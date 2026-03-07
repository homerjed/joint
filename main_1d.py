import argparse
import inspect
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
from joint.flow import SolutionKwargs, get_time_embedder, single_sample_fn_ode, single_sample_modalities_fn_ode
from joint.utils import (
    ModalitySpec,
    ModalityInfos,
    modality_dataloader,
    count_parameters,
    make_layernorm_decay_mask,
)
from joint.train import make_step, evaluate_validation_set, save_training_state, load_training_state


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="0_1d",
        help="Run tag used as a subdirectory inside `out/`",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload config/train_state from `out/<run_name>/` if present (use --no-reload for fresh run).",
    )
    args = parser.parse_args()
    return args


def grf1d_powerlaw(
    key,
    A: float,
    B: float,
    n: int,
    L: float,
    k0: float = 1.0,
    mean: float = 0.0,
    zero_mode: float = 0.0,
    eps: float = 1e-12,
):
    k = 2.0 * jnp.pi * jnp.fft.rfftfreq(n, d=L / n)
    Pk = A * jnp.power((jnp.maximum(k, eps) / k0), B)

    key_re, key_im = jr.split(key, 2)
    re = jr.normal(key_re, shape=Pk.shape)
    im = jr.normal(key_im, shape=Pk.shape)
    F = (re + 1j * im) * jnp.sqrt(Pk / 2.0)
    F = F.at[0].set(zero_mode + 0.0j)
    if n % 2 == 0:
        F = F.at[-1].set(jnp.real(F[-1]) + 0.0j)

    field = jnp.fft.irfft(F, n=n)
    field = field * jnp.sqrt(n)
    return field + mean


if __name__ == "__main__":

    # TODO: make sample fns a method of FlowTransformer, so whole framework can be agnostic to type of 
    # generative model that is being used to train the transformer

    # TODO: allow for lognormal fields
 
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
    image_shape = (config.data_config.n_channels, config.data_config.n_pix)
    parameter_shape = tuple(config.data_config.parameter_shape)
    num_patches = config.data_config.n_pix // config.model_config.patch_size

    # Other
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
        field = grf1d_powerlaw(
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
    X = X[:, jnp.newaxis, :]
    X_mean = jnp.mean(X, axis=0)
    X_std = jnp.std(X, axis=0)
    X = (X - X_mean) / X_std

    Y = jnp.stack([A, B], axis=1)
    n_samples = len(X)
    n_valid = max(1, int(n_samples * val_fraction))
    n_valid = min(n_valid, n_samples - 1)
    n_train = n_samples - n_valid

    # Dataloaders
    key, key_split = jr.split(key)
    idx = jr.permutation(key_split, n_samples)
    train_idx = idx[:n_train]
    valid_idx = idx[n_train:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_valid, Y_valid = X[valid_idx], Y[valid_idx]

    # Fixed conditioning target for conditional sampling diagnostics.
    key, key_ref_idx = jr.split(key)
    ref_idx = int(jr.randint(key_ref_idx, shape=(), minval=0, maxval=len(X_train)))
    x_ref_train = X_train[ref_idx]
    a_ref = Y_train[ref_idx, 0]
    b_ref = Y_train[ref_idx, 1]

    n_batch_train = max(1, int(config.train_config.n_batch))
    n_batch_valid = max(1, int(getattr(config.train_config, "n_batch_valid", n_batch_train)))
    n_batch_valid = min(n_batch_valid, len(X_valid))
    n_valid_batches = max(1, len(X_valid) // n_batch_valid)

    key, key_train_loader, key_valid_loader = jr.split(key, 3)

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
            "x": X_valid, 
            "a": Y_valid[:, 0][:, jnp.newaxis], 
            "b": Y_valid[:, 1][:, jnp.newaxis]
        },
        n_batch=n_batch_valid,
        key=key_valid_loader,
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

    print("Model has {:.3E} parameters.".format(count_parameters(v)))

    # Optimiser, EMA and fixed sampling noise
    opt_kwargs = dict(config.train_config.opt_kwargs)
    try:
        factory_params = inspect.signature(config.train_config.opt_factory).parameters
    except (TypeError, ValueError):
        factory_params = {}
    if "mask" in factory_params and "mask" not in opt_kwargs:
        opt_kwargs["mask"] = make_layernorm_decay_mask(v)
    opt = config.train_config.opt_factory(**opt_kwargs)

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
    last_valid_loss = None
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
                    n_valid_batches,
                    key=key_eval,
                    solution_kwargs=solution_kwargs,
                )
                last_valid_loss = float(val_loss)
                val_steps.append(step)
                val_losses.append(last_valid_loss)

            if last_valid_loss is None:
                steps.set_postfix(L=float(loss))
            else:
                steps.set_postfix(L=float(loss), L_valid=last_valid_loss)

            if step % sample_every == 0:

                # Sample modalities 
                sample_fn = lambda z: single_sample_fn_ode(v, z, solution_kwargs)
                samples_dict = jax.vmap(sample_fn)(Z)
                generated_samples = samples_dict["x"][:, 0, :]

                train_idx = jr.choice(key, len(X_train), shape=(sample_n_samples ** 2,))
                train_samples = X_train[train_idx, 0, :]

                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                ax = axs[0]
                ax.imshow(
                    generated_samples,
                    cmap="coolwarm",
                    vmin=X_train.min(),
                    vmax=X_train.max(),
                    aspect="auto",
                )
                ax.set_title("Generated x")
                ax = axs[1]
                ax.imshow(
                    train_samples,
                    cmap="coolwarm",
                    vmin=X_train.min(),
                    vmax=X_train.max(),
                    aspect="auto",
                )
                ax.set_title("Train x")
                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.tight_layout()
                fig.savefig(samples_dir / f"samples_{step:07d}.png")
                plt.close(fig)

                # Sample x with fixed parameters a,b.
                Z_fixed_ab = {
                    name: jr.normal(jr.fold_in(key, i), (sample_n_samples ** 2, *spec.input_shape))
                    for i, (name, spec) in enumerate(modality_infos.items())
                }
                Z_fixed_ab["a"] = jnp.full((sample_n_samples ** 2, 1), a_ref)
                Z_fixed_ab["b"] = jnp.full((sample_n_samples ** 2, 1), b_ref)

                key, key_cond_ab = jr.split(key)
                cond_keys_ab = jr.split(key_cond_ab, sample_n_samples ** 2)
                sample_fn_fixed_ab = lambda z, k: single_sample_modalities_fn_ode(
                    v, 
                    z, 
                    modality_infos=modality_infos, 
                    modalities_to_sample=["x"], 
                    solution_kwargs=solution_kwargs,
                    key=k,
                    bridge_path=path,
                )
                samples_fixed_ab = jax.vmap(sample_fn_fixed_ab)(Z_fixed_ab, cond_keys_ab)
                generated_x_fixed_ab = samples_fixed_ab["x"][:, 0, :]

                # Sample a,b with fixed x ~ N(0, 1).
                key, key_x_fixed = jr.split(key)
                x_fixed = X_valid[:1] # jr.normal(key_x_fixed, (1, *modality_infos["x"].input_shape))
                Z_fixed_x = {
                    name: jr.normal(jr.fold_in(key, i + 100), (sample_n_samples ** 2, *spec.input_shape))
                    for i, (name, spec) in enumerate(modality_infos.items())
                }
                Z_fixed_x["x"] = jnp.repeat(x_fixed, sample_n_samples ** 2, axis=0)

                key, key_cond_x = jr.split(key)
                cond_keys_x = jr.split(key_cond_x, sample_n_samples ** 2)
                sample_fn_fixed_x = lambda z, k: single_sample_modalities_fn_ode(
                    v,
                    z,
                    modality_infos=modality_infos,
                    modalities_to_sample=["a", "b"],
                    solution_kwargs=solution_kwargs,
                    key=k,
                    bridge_path=path,
                )
                samples_fixed_x = jax.vmap(sample_fn_fixed_x)(Z_fixed_x, cond_keys_x)
                generated_a_grid = rearrange(
                    samples_fixed_x["a"],
                    "(p q) 1 -> p q",
                    p=sample_n_samples
                )
                generated_b_grid = rearrange(
                    samples_fixed_x["b"],
                    "(p q) 1 -> p q",
                    p=sample_n_samples
                )

                fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                ax = axs[0, 0]
                ax.imshow(
                    generated_x_fixed_ab,
                    cmap="coolwarm",
                    vmin=X_train.min(),
                    vmax=X_train.max(),
                    aspect="auto",
                )
                ax.set_title(f"Generated x | fixed a,b=({float(a_ref):.2f},{float(b_ref):.2f})")
                ax = axs[0, 1]
                ax.plot(x_ref_train[0], color="black", linewidth=1.5)
                ax.set_ylim(float(X_train.min()), float(X_train.max()))
                ax.set_title("Real train x (same a,b)")

                ax = axs[1, 0]
                ax.imshow(generated_a_grid, cmap="viridis", vmin=Y_train[:, 0].min(), vmax=Y_train[:, 0].max())
                ax.set_title("Generated a | fixed x~N(0,1)")
                ax = axs[1, 1]
                ax.imshow(generated_b_grid, cmap="viridis", vmin=Y_train[:, 1].min(), vmax=Y_train[:, 1].max())
                ax.set_title("Generated b | fixed x~N(0,1)")

                for ax_row in axs:
                    for ax in ax_row:
                        ax.set_xticks([])
                        ax.set_yticks([])
                fig.tight_layout()
                fig.savefig(samples_dir / f"samples_fixed_ab_{step:07d}.png")
                plt.close(fig)

                # Losses
                plt.figure()
                plt.loglog(train_steps, train_losses, label="train")
                if val_losses:
                    plt.loglog(val_steps, val_losses, label="valid")
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "loss.png")
                plt.close()

                # Save train state
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
