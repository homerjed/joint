import argparse
import time
from copy import deepcopy
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt
from tqdm.auto import trange
from einops import rearrange

from joint.config import get_config
from joint.modules import ModalitiesEncoder, ModalitiesDecoder
from joint.transformer import VisionTransformer, FlowTransformer
from joint.flow import SolutionKwargs, get_time_embedder, single_sample_fn_ode
from joint.utils import ModalitySpec, ModalityInfos, grf2d_powerlaw, modality_dataloader
from joint.train import make_step, evaluate_validation_set, save_training_state, load_training_state


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="0",
        help="Run tag used as a subdirectory inside out/",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload config/train_state from out/<run_name>/ if present (use --no-reload for fresh run).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
 
    args = get_args()

    out_root = Path("out")
    out_dir = out_root / args.run_name
    samples_dir = out_dir / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    train_state_path = out_dir / "train_state.eqx"

    config = get_config(config_path, reload=args.reload)

    key = jr.key(config.seed)

    # Data shapes
    image_shape = (config.data_config.n_channels, config.data_config.n_pix, config.data_config.n_pix)
    parameter_shape = tuple(config.data_config.parameter_shape)
    num_patches = (config.data_config.n_pix // config.model_config.patch_size) ** 2

    # Other
    opt = config.train_config.opt_factory(**config.train_config.opt_kwargs)
    use_ema = config.train_config.use_ema
    sample_every = config.train_config.sample_every
    n_steps_validate = max(1, int(getattr(config.train_config, "n_steps_validate", sample_every)))
    val_fraction = float(getattr(config.data_config, "val_fraction", 0.1))
    sample_n_samples = config.sampling_config.n_samples

    # Generate dataset
    key_A, key_B = jr.split(key)
    A = jr.uniform(key_A, (config.data_config.n_samples,), minval=1.0, maxval=4.0)
    B = jr.uniform(key_B, (config.data_config.n_samples,), minval=1.0, maxval=4.0) * -1.

    def generator(key, A, B): 
        field = grf2d_powerlaw(
            key, 
            n=config.data_config.n_pix, 
            L=1.0, 
            A=A, 
            B=B, 
            k0=2.0 * jnp.pi
        )
        return field

    keys = jr.split(key, len(A))
    X = jax.vmap(generator)(keys, A, B)
    X = X[:, jnp.newaxis, :, :]
    X = (X - jnp.mean(X, axis=0)) / jnp.std(X, axis=0)

    Y = jnp.stack([A, B], axis=1)
    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to create train/validation splits.")
    n_val = max(1, int(n_samples * val_fraction))
    n_val = min(n_val, n_samples - 1)
    n_train = n_samples - n_val

    # Dataloaders
    key, key_split = jr.split(key)
    idx = jr.permutation(key_split, n_samples)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    n_batch_train = max(1, int(config.train_config.n_batch))
    n_batch_val = max(1, int(getattr(config.train_config, "n_batch_val", n_batch_train)))
    n_batch_val = min(n_batch_val, len(X_val))
    n_val_batches = max(1, len(X_val) // n_batch_val)

    key, key_train_loader, key_val_loader = jr.split(key, 3)

    train_loader = modality_dataloader(
        {
            "x": X_train, 
            "a": Y_train[:, 0][:, jnp.newaxis], 
            "b": Y_train[:, 1][:, jnp.newaxis]
        },
        n_batch=n_batch_train,
        key=key_train_loader,
    )
    val_loader = modality_dataloader(
        {
            "x": X_val, 
            "a": Y_val[:, 0][:, jnp.newaxis], 
            "b": Y_val[:, 1][:, jnp.newaxis]
        },
        n_batch=n_batch_val,
        key=key_val_loader,
        shuffle=False,
    )

    # Modality information
    modality_infos = ModalityInfos(
        {
            "x": ModalitySpec(
                name="x",
                modality_type="image",
                input_shape=image_shape,
                patch_size=config.model_config.patch_size,
                sequence_length=num_patches,
                embedding_dim=config.model_config.embedding_dim,
            ),
            "a": ModalitySpec(
                name="a",
                modality_type="vector",
                input_shape=(1,),
                sequence_length=1,
                embedding_dim=config.model_config.embedding_dim,
            ),
            "b": ModalitySpec(
                name="b",
                modality_type="vector",
                input_shape=(1,),
                sequence_length=1,
                embedding_dim=config.model_config.embedding_dim,
            ),
        }
    )

    # Multimodal model
    key_encoder, key_decoder, key_transformer = jr.split(key, 3)
    
    # > Encoder and decoder
    modalities_encoder = ModalitiesEncoder(modality_infos, key=key_encoder)

    modalities_decoder = ModalitiesDecoder(modality_infos, key=key_decoder)

    # > DiT-like transformer
    transformer = VisionTransformer(
        embedding_dim=config.model_config.embedding_dim,
        hidden_dim=config.model_config.hidden_dim,
        num_heads=config.model_config.n_heads,
        num_layers=config.model_config.n_blocks,
        dropout_rate=config.model_config.dropout_rate,
        key=key_transformer
    )

    # > Flow path, diffusion-time embedder and flow-wrapper
    path = config.sampling_config.path_factory()
    sde_name = path.__class__.__name__.removesuffix("Path").lower()

    solution_kwargs = SolutionKwargs(
        config.sampling_config.solver_factory(), 
        config.sampling_config.t0, 
        config.sampling_config.t1, 
        config.sampling_config.dt0
    )

    time_embedder = get_time_embedder(config.model_config.embedding_dim)

    v = FlowTransformer(
        modalities_encoder, 
        modalities_decoder, 
        transformer,
        time_embedder=time_embedder,
        modality_infos=modality_infos
    )

    # Optimiser, EMA and fixed sampling noise
    opt_state = opt.init(eqx.filter(v, eqx.is_inexact_array))

    if use_ema:
        v_ema = deepcopy(v)
    else:
        v_ema = None

    # Dict of ALL modalities for sampling
    Z = {
        name: jr.normal(jr.fold_in(key, i), (sample_n_samples ** 2, *spec.input_shape))
        for i, (name, spec) in enumerate(modality_infos.items())
    }

    start_step = 0
    if args.reload and train_state_path.exists():
        v, v_ema, opt, opt_state, start_step = load_training_state(
            train_state_path,
            model=v,
            optimiser=opt,
            optimiser_state=opt_state,
        )

    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    last_val_loss = None
    with trange(
        start_step, 
        config.train_config.n_steps, 
        colour="green"
    ) as steps:
        for step in steps:
            key, key_step = jr.split(key)
            batch = next(train_loader)

            v, loss, opt_state, v_ema = make_step(
                v, 
                path, 
                key_step, 
                batch,
                opt_state, 
                opt=opt,
                v_ema=v_ema, 
                solution_kwargs=solution_kwargs
            )

            train_steps.append(step)
            train_losses.append(float(loss))

            if step % n_steps_validate == 0:
                key, key_eval = jr.split(key)
                val_loss = evaluate_validation_set(
                    v_ema if v_ema is not None else v,
                    path,
                    val_loader,
                    n_val_batches,
                    key=key_eval,
                    solution_kwargs=solution_kwargs,
                )
                last_val_loss = float(val_loss)
                val_steps.append(step)
                val_losses.append(last_val_loss)

            if last_val_loss is None:
                steps.set_postfix(L=float(loss))
            else:
                steps.set_postfix(L=float(loss), L_val=last_val_loss)

            if step % sample_every == 0:
                samples_dict = jax.vmap(lambda z: single_sample_fn_ode(v, z, solution_kwargs))(Z)
                generated_samples = rearrange(
                    samples_dict["x"], 
                    "(p q) c h w -> (p h) (q w) c", 
                    p=sample_n_samples
                )

                train_idx = jr.choice(key, len(X_train), shape=(sample_n_samples ** 2,))
                train_samples = rearrange(
                    X_train[train_idx],
                    "(p q) c h w -> (p h) (q w) c",
                    p=sample_n_samples
                )

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                ax = axs[0]
                ax.imshow(generated_samples, cmap="coolwarm", vmin=X_train.min(), vmax=X_train.max())
                ax = axs[1]
                ax.imshow(train_samples, cmap="coolwarm", vmin=X_train.min(), vmax=X_train.max())
                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.tight_layout()
                fig.savefig(samples_dir / f"samples_{step:07d}.png")
                plt.close(fig)

                plt.figure()
                plt.semilogy(train_steps, train_losses, label="train")
                if val_losses:
                    plt.semilogy(val_steps, val_losses, label="valid")
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "loss.png")
                plt.close()

                save_training_state(
                    train_state_path,
                    v, 
                    v_ema,
                    opt,
                    opt_state,
                    step
                )

    if v_ema is not None:
        v = v_ema
