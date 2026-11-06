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

from joint.config import (
    MODEL_FAMILY_CHOICES,
    get_config,
    save_json_config,
    set_model_family,
)
from joint.generative import build_model_family
from joint.utils import (
    ModalitySpec,
    ModalityInfos,
    modality_dataloader,
    count_parameters,
)
from joint.train import (
    build_make_step,
    build_evaluate,
    evaluate_validation_batches,
    save_training_state,
    load_training_state,
    get_optimiser,
)

try:
    from scripts.random_fields import grf2d_powerlaw
except ImportError:
    from random_fields import grf2d_powerlaw


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_name",
        type=str,
        default="0_2d",
        help="Run tag used as a subdirectory inside `out/`",
    )
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload config/train_state from `out/<run_name>/` if present (use --no-reload for fresh run).",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=MODEL_FAMILY_CHOICES,
        default=None,
        help="Override `config.model_config.model_family` for this run and persist it to `out/<run_name>/config.json`.",
    )
    parser.add_argument(
        "--flow_map_loss", choices=("lsd", "esd", "psd"), default=None,
        help="Self-distillation loss for the flow_map family (default: lsd).",
    )
    parser.add_argument(
        "--flow_map_steps", type=int, default=None,
        help="Number of learned-map sampling steps (default: 4; use 1 for one-step).",
    )
    parser.add_argument(
        "--flow_map_psd_split", choices=("uniform", "midpoint"), default=None,
        help="Intermediate-time distribution for PSD.",
    )
    args = parser.parse_args()
    if args.flow_map_steps is not None and args.flow_map_steps < 1:
        parser.error("--flow_map_steps must be positive.")
    return args


if __name__ == "__main__":
    # TODO: try inpainting with a masked image
 
    args = get_args()

    out_root = Path("out")
    out_dir = out_root / args.run_name
    samples_dir = out_dir / "samples"
    samples_grid_dir = samples_dir / "samples"
    samples_fixed_ab_dir = samples_dir / "samples_fixed_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_grid_dir.mkdir(parents=True, exist_ok=True)
    samples_fixed_ab_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    train_state_path = out_dir / "train_state.eqx"

    config = get_config(config_path, reload=args.reload)
    previous_model_family = str(config.model_config.model_family)
    if args.model_family is not None:
        requested_model_family = str(args.model_family)
        if (
            args.reload
            and train_state_path.exists()
            and previous_model_family != requested_model_family
        ):
            raise ValueError(
                "Cannot switch model_family from "
                f"{previous_model_family!r} to {requested_model_family!r} "
                f"while reloading {train_state_path}. Use a new --run_name or start without --reload."
            )
        if previous_model_family != requested_model_family:
            config = set_model_family(config, requested_model_family)
            save_json_config(config_path, config)
    flow_map_overrides = (
        args.flow_map_loss, args.flow_map_steps, args.flow_map_psd_split
    )
    if any(value is not None for value in flow_map_overrides):
        if config.model_config.model_family != "flow_map":
            raise ValueError("Flow-map options require --model_family flow_map.")
        if args.flow_map_loss is not None:
            config.train_config.flow_map_loss = args.flow_map_loss
        if args.flow_map_steps is not None:
            config.sampling_config.flow_map_steps = args.flow_map_steps
        if args.flow_map_psd_split is not None:
            config.train_config.flow_map_psd_split = args.flow_map_psd_split
        save_json_config(config_path, config)
    backbone_config = config.model_config.backbone

    key = jr.key(config.seed)

    # Data shapes
    image_shape = (config.data_config.n_channels, config.data_config.n_pix, config.data_config.n_pix)
    parameter_shape = tuple(config.data_config.parameter_shape)
    num_patches = (config.data_config.n_pix // backbone_config.patch_size) ** 2

    # Other
    use_ema = config.train_config.use_ema
    sample_every = config.train_config.sample_every
    n_steps_validate = max(1, int(getattr(config.train_config, "n_steps_validate", sample_every)))
    val_fraction = float(getattr(config.data_config, "val_fraction", 0.1))
    sample_n_samples = config.sampling_config.n_samples
    time_sampling_distribution = str(getattr(config.train_config, "time_sampling_distribution", "uniform"))
    time_sampling_beta_alpha = float(getattr(config.train_config, "time_sampling_beta_alpha", 1.0))
    time_sampling_beta_beta = float(getattr(config.train_config, "time_sampling_beta_beta", 1.0))
    time_sampling_stratified = bool(getattr(config.train_config, "time_sampling_stratified", True))
    n_t_bins = max(1, int(getattr(config.train_config, "n_t_bins", 4)))
    conditional_training_probability = float(
        getattr(config.train_config, "conditional_training_probability", 0.0)
    )
    t_bin_edges = jnp.linspace(
        config.sampling_config.t0, config.sampling_config.t1, n_t_bins + 1
    )

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
    X_mean = jnp.mean(X, axis=0)
    X_std = jnp.std(X, axis=0)
    X = (X - X_mean) / X_std

    Y_raw = jnp.stack([A, B], axis=1)
    Y_mean = jnp.mean(Y_raw, axis=0, keepdims=True)
    Y_std = jnp.std(Y_raw, axis=0, keepdims=True)
    Y_std = jnp.where(Y_std > 0.0, Y_std, 1.0)
    Y = (Y_raw - Y_mean) / Y_std
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
    Y_train_raw, Y_valid_raw = Y_raw[train_idx], Y_raw[valid_idx]
    x_plot_vmin, x_plot_vmax = [
        float(v) for v in jnp.percentile(X_train, jnp.array([1.0, 99.0]))
    ]

    # Fixed conditioning target for conditional sampling diagnostics.
    key, key_ref_idx = jr.split(key)
    ref_idx = int(jr.randint(key_ref_idx, shape=(), minval=0, maxval=len(X_train)))
    x_ref_train = X_train[ref_idx]
    a_ref = Y_train[ref_idx, 0]
    b_ref = Y_train[ref_idx, 1]
    a_ref_raw = a_ref * Y_std[0, 0] + Y_mean[0, 0]
    b_ref_raw = b_ref * Y_std[0, 1] + Y_mean[0, 1]

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
                patch_size=backbone_config.patch_size,
                sequence_length=num_patches,
                embedding_dim=backbone_config.embedding_dim,
            ),
            "a": ModalitySpec(
                name="a",
                modality_type="vector",
                input_shape=(1,),
                sequence_length=1,
                embedding_dim=backbone_config.embedding_dim,
            ),
            "b": ModalitySpec(
                name="b",
                modality_type="vector",
                input_shape=(1,),
                sequence_length=1,
                embedding_dim=backbone_config.embedding_dim,
            ),
        }
    )

    # Multimodal generative family
    key, key_model = jr.split(key)
    family = build_model_family(config, modality_infos, key_model)
    v = family.model
    make_step = build_make_step(family.loss_fn)
    evaluate = build_evaluate(family.loss_fn)

    print("Model has {:.3E} parameters.".format(count_parameters(v)))

    # Optimiser, EMA and fixed sampling noise
    opt = get_optimiser(v, config.train_config)
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
            use_ema=use_ema,
            model=v,
            optimiser=opt,
            optimiser_state=opt_state,
        )

    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    train_modality_losses = {name: [] for name in sorted(modality_infos.keys())}
    train_t_bin_losses = []
    last_valid_loss = None
    with trange(
        start_step, 
        config.train_config.n_steps, 
        colour="green"
    ) as steps:
        for step in steps:
            key, key_step = jr.split(key)
            batch = next(train_loader)

            v, loss, opt_state, v_ema, train_metrics = make_step(
                v,
                key_step,
                batch,
                opt_state,
                opt=opt,
                ema_model=v_ema,
            )

            train_steps.append(step)
            train_losses.append(float(loss))
            for name in train_modality_losses:
                train_modality_losses[name].append(float(train_metrics["per_modality_loss"][name]))
            if "t_bin_loss" in train_metrics:
                train_t_bin_losses.append(jnp.asarray(train_metrics["t_bin_loss"]))

            if step % n_steps_validate == 0:
                key, key_eval = jr.split(key)
                val_loss = evaluate_validation_batches(
                    evaluate,
                    v_ema if v_ema is not None else v,
                    val_loader,
                    n_valid_batches,
                    key=key_eval,
                )
                last_valid_loss = float(val_loss)
                val_steps.append(step)
                val_losses.append(last_valid_loss)

            postfix = {"L": float(loss)}
            if last_valid_loss is not None:
                postfix["L_valid"] = last_valid_loss
            for name in sorted(train_modality_losses.keys()):
                postfix[f"L_{name}"] = train_modality_losses[name][-1]
            if train_t_bin_losses:
                postfix["L_tmid"] = float(train_t_bin_losses[-1][n_t_bins // 2])
            steps.set_postfix(**postfix)

            if step % sample_every == 0:
                model_for_sampling = v_ema if v_ema is not None else v

                # Sample modalities 
                key, key_sample = jr.split(key)
                sample_keys = jr.split(key_sample, sample_n_samples ** 2)
                sample_fn = lambda z, sample_key: family.sample_fn(
                    model_for_sampling,
                    z,
                    sample_key,
                )
                samples_dict = jax.vmap(sample_fn)(Z, sample_keys)
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
                ax.imshow(
                    generated_samples,
                    cmap="coolwarm",
                    vmin=x_plot_vmin,
                    vmax=x_plot_vmax,
                )
                ax = axs[1]
                ax.imshow(
                    train_samples,
                    cmap="coolwarm",
                    vmin=x_plot_vmin,
                    vmax=x_plot_vmax,
                )
                for ax in axs:
                    ax.set_xticks([])
                    ax.set_yticks([])
                fig.tight_layout()
                fig.savefig(samples_grid_dir / f"samples_{step:07d}.png")
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
                sample_fn_fixed_ab = lambda z, sample_key: family.sample_modalities_fn(
                    model_for_sampling,
                    z, 
                    ["x"],
                    sample_key,
                )
                samples_fixed_ab = jax.vmap(sample_fn_fixed_ab)(Z_fixed_ab, cond_keys_ab)
                generated_x_fixed_ab = rearrange(
                    samples_fixed_ab["x"], 
                    "(p q) c h w -> (p h) (q w) c", 
                    p=sample_n_samples
                )

                # Sample a,b with fixed x.
                x_fixed = X_valid[:1]
                Z_fixed_x = {
                    name: jr.normal(jr.fold_in(key, i + 100), (sample_n_samples ** 2, *spec.input_shape))
                    for i, (name, spec) in enumerate(modality_infos.items())
                }
                Z_fixed_x["x"] = jnp.repeat(x_fixed, sample_n_samples ** 2, axis=0)

                key, key_cond_x = jr.split(key)
                cond_keys_x = jr.split(key_cond_x, sample_n_samples ** 2)
                sample_fn_fixed_x = lambda z, sample_key: family.sample_modalities_fn(
                    model_for_sampling,
                    z,
                    ["a", "b"],
                    sample_key,
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
                generated_a_grid_raw = generated_a_grid * Y_std[0, 0] + Y_mean[0, 0]
                generated_b_grid_raw = generated_b_grid * Y_std[0, 1] + Y_mean[0, 1]

                fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                ax = axs[0, 0]
                ax.imshow(
                    generated_x_fixed_ab,
                    cmap="coolwarm",
                    vmin=x_plot_vmin,
                    vmax=x_plot_vmax,
                )
                ax.set_title(
                    f"Generated x | fixed a,b=({float(a_ref_raw):.2f},{float(b_ref_raw):.2f})"
                )
                ax = axs[0, 1]
                ax.imshow(
                    jnp.transpose(x_ref_train, (1, 2, 0)),
                    cmap="coolwarm",
                    vmin=x_plot_vmin,
                    vmax=x_plot_vmax,
                )
                ax.set_title("Real train x (same a,b)")

                ax = axs[1, 0]
                ax.imshow(
                    generated_a_grid_raw,
                    cmap="viridis",
                    vmin=Y_train_raw[:, 0].min(),
                    vmax=Y_train_raw[:, 0].max(),
                )
                ax.set_title("Generated a (raw) | fixed x")
                ax = axs[1, 1]
                ax.imshow(
                    generated_b_grid_raw,
                    cmap="viridis",
                    vmin=Y_train_raw[:, 1].min(),
                    vmax=Y_train_raw[:, 1].max(),
                )
                ax.set_title("Generated b (raw) | fixed x")

                for ax_row in axs:
                    for ax in ax_row:
                        ax.set_xticks([])
                        ax.set_yticks([])
                fig.tight_layout()
                fig.savefig(samples_fixed_ab_dir / f"samples_fixed_ab_{step:07d}.png")
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

                plt.figure()
                for name in sorted(train_modality_losses.keys()):
                    plt.loglog(train_steps, train_modality_losses[name], label=f"train_{name}")
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "loss_by_modality.png")
                plt.close()

                if train_t_bin_losses:
                    plt.figure()
                    t_bin_matrix = jnp.stack(train_t_bin_losses)
                    for i in range(n_t_bins):
                        bin_label = (
                            f"[{float(t_bin_edges[i]):.2f},"
                            f"{float(t_bin_edges[i + 1]):.2f}{']' if i == n_t_bins - 1 else ')'}"
                        )
                        plt.loglog(train_steps, t_bin_matrix[:, i], label=bin_label)
                    plt.xlabel("step")
                    plt.ylabel("loss")
                    plt.legend(title="t bin")
                    plt.tight_layout()
                    plt.savefig(out_dir / "loss_vs_t_bins.png")
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
