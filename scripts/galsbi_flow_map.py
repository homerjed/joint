"""Train joint GalSBI image, galaxy-set and population-parameter flow maps.

We learn empirical P(N) * P(image, catalog, parameters | N): sample a training
count, then generate all three other modalities from independent noise.
Catalog padding is masked; binary types use fresh dequantization and a logit.
All fitted normalization uses the training split, grouped by model_index.

Boffi et al. (2025): https://arxiv.org/abs/2505.18825
Outputs include checkpoints, normalization, generated NPZs and PNG diagnostics.
A short run checks the pipeline, not scientific fidelity. Generated population
parameters do not enforce positivity, prior bounds or simplex constraints.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joint.config import get_config, save_json_config, set_model_family
from joint.generative import build_model_family
from joint.generative.flow_map.sampling import single_sample_modalities_fn
from joint.train import load_training_state, save_training_state
from joint.utils import ModalityInfos, ModalitySpec, apply_ema, count_parameters

EPS = 1e-5


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train and sample a joint image/catalog/parameter GalSBI flow map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, help="Input data.npz; otherwise select the latest run under --data-dir.")
    parser.add_argument("--data-dir", type=Path, default=Path("out/galsbi"), help="Search directory for */data.npz.")
    parser.add_argument("--output-dir", type=Path, default=Path("out/galsbi_flow_map"), help="Parent of a new timestamped run directory.")
    parser.add_argument("--steps", type=int, default=1000, help="Training updates.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training minibatch size.")
    parser.add_argument("--loss", choices=("psd", "esd", "lsd"), default="psd", help="Self-distillation objective; ESD/LSD use JVPs.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--embedding-dim", type=int, default=64, help="Token embedding width.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Transformer MLP width.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--blocks", type=int, default=3, help="Transformer blocks.")
    parser.add_argument("--patch-size", type=int, default=4, help="Square image patch side in pixels.")
    parser.add_argument("--n-generate", type=int, default=8, help="Number of joint generated samples.")
    parser.add_argument("--sampling-steps", type=int, nargs="+", default=[1, 4, 8], help="Compare map compositions using identical base noise/counts.")
    parser.add_argument("--log-every", type=int, default=50, help="Updates between diagnostic logs.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Maximum gradient global norm.")
    parser.add_argument("--ema-rate", type=float, default=0.999, help="EMA retention rate in [0, 1).")
    parser.add_argument("--validation-fraction", type=float, default=0.2, help="Fraction of model-index groups held out.")
    args = parser.parse_args(argv)
    sizes = [args.steps, args.batch_size, args.embedding_dim, args.hidden_dim,
             args.heads, args.blocks, args.patch_size, args.n_generate, args.log_every]
    if min(sizes + args.sampling_steps) < 1 or args.seed < 0:
        parser.error("sizes and step counts must be positive; seed must be nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.grad_clip)):
        parser.error("--learning-rate and --grad-clip must be finite and positive")
    if args.embedding_dim % args.heads:
        parser.error("--embedding-dim must be divisible by --heads")
    if not math.isfinite(args.ema_rate) or not 0 <= args.ema_rate < 1:
        parser.error("--ema-rate must be finite and in [0, 1)")
    if not math.isfinite(args.validation_fraction) or not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be finite and between 0 and 1")
    return args


def logit(probability):
    return np.log(probability) - np.log1p(-probability)


def safe_scale(value):
    return np.where(value > 1e-6, value, 1.0).astype(np.float32)


def standardize(values, training, name, stats, *, axis=0, keepdims=False):
    """Fit on training values, retain the inverse, and handle empty/constant features."""
    if len(training):
        mean = training.mean(axis=axis, keepdims=keepdims)
        scale = safe_scale(training.std(axis=axis, keepdims=keepdims))
    else:
        mean = np.zeros(values.shape[-1], np.float32)
        scale = np.ones(values.shape[-1], np.float32)
    stats[f"{name}_mean"], stats[f"{name}_scale"] = mean, scale
    return ((values - mean) / scale).astype(np.float32)


def catalog_coordinates(raw, mask, feature, width, height, *, inverse=False):
    """Map positions/positive features/ellipticities to R, or invert that map."""
    rows = np.where(mask[..., None], raw, 0.0).astype(np.float64)
    valid = rows[mask]
    for name, size in (("x", width), ("y", height)):
        column = feature[name]
        if inverse:
            rows[..., column] = size * (0.5 + 0.5 * np.tanh(rows[..., column] / 2.0))
        else:
            if np.any((valid[:, column] < 0) | (valid[:, column] > size)):
                raise ValueError(f"Catalog {name} lies outside the image footprint")
            rows[..., column] = logit(np.clip(rows[..., column] / size, EPS, 1.0 - EPS))
    for name in ("z", "r50", "sersic_n"):
        column = feature[name]
        if not inverse and np.any(valid[:, column] < 0):
            raise ValueError(f"Negative intrinsic {name}")
        rows[..., column] = np.exp(rows[..., column]) if inverse else np.log(np.maximum(rows[..., column], EPS))
    columns = [feature["e1"], feature["e2"]]
    e = rows[..., columns]
    if inverse:
        rows[..., columns] = e / (1.0 + np.hypot(e[..., 0], e[..., 1])[..., None])
        rows[..., feature["galaxy_type"]] = rows[..., feature["galaxy_type"]] >= 0.0
    else:
        radius = np.linalg.norm(e, axis=-1, keepdims=True)
        if np.any(radius[..., 0][mask] >= 1):
            raise ValueError("Ellipticity must lie inside the unit disk")
        rows[..., columns] = e / (1.0 - radius)
        rows[..., feature["galaxy_type"]] = 0.0  # Fresh dequantization is added per batch.
    return np.where(mask[..., None], rows, 0.0).astype(np.float32)


def decode_samples(samples, mask, stats, feature, width, height):
    """Invert normalization; population-parameter constraints are not imposed."""
    image = stats["image_softening"] * np.sinh(
        samples["image"].astype(np.float64) * stats["image_scale"] + stats["image_mean"]
    )
    parameters = samples["parameters"].astype(np.float64) * stats["parameter_scale"] + stats["parameter_mean"]
    rows = samples["catalog"].astype(np.float64) * stats["catalog_scale"] + stats["catalog_mean"]
    rows = catalog_coordinates(rows, mask, feature, width, height, inverse=True)
    result = {
        "images": image.astype(np.float32), "catalog": rows,
        "parameters": parameters.astype(np.float32),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("Nonfinite decoded samples; inspect training stability before using this model.")
    return result


def plot_history(history, loss_type, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for name in ("train", "validation", "flow_matching", "distillation"):
        axes[0].plot([r["step"] for r in history], [r[name] for r in history], marker=".", label=name)
    for name in ("image", "catalog", "parameters"):
        axes[1].plot([r["step"] for r in history], [r[f"loss_{name}"] for r in history], marker=".", label=name)
    axes[0].set_title(f"{loss_type.upper()} objective diagnostics")
    axes[1].set_title("Per-modality training losses")
    for axis in axes:
        axis.set(xlabel="Training step", ylabel="Loss")
        axis.legend()
    fig.savefig(run_dir / "training_loss.png", dpi=150)
    plt.close(fig)


def plot_samples(generated, generated_mask, generated_counts, bands, feature,
                 sampling_steps, validation_parameters, parameter_names, count_probability, run_dir):
    """Compare common-noise samples, source positions, counts and held-out parameters."""
    height, width = next(iter(generated.values()))["images"].shape[-2:]
    capacity = generated_mask.shape[1]
    count_values = np.arange(capacity + 1)
    i_band = bands.index("i") if "i" in bands else 0
    n_display = min(3, len(generated_counts))
    fig, axes = plt.subplots(n_display, len(sampling_steps), figsize=(4 * len(sampling_steps), 3.5 * n_display),
                             squeeze=False, layout="constrained")
    for row in range(n_display):
        for col, steps in enumerate(sampling_steps):
            sample = generated[steps]
            image = sample["images"][row, i_band]
            valid_rows = sample["catalog"][row, generated_mask[row]]
            lo, hi = np.percentile(image, [1, 99.5])
            axes[row, col].imshow(image, origin="lower", extent=(0, width, 0, height),
                                  cmap="gray", vmin=lo, vmax=hi)
            axes[row, col].scatter(valid_rows[:, feature["x"]], valid_rows[:, feature["y"]],
                                   marker="+", s=35, c="tab:orange")
            axes[row, col].set(title=f"{steps} steps; N={generated_counts[row]}", xlabel="x [pixel]", ylabel="y [pixel]")
    fig.savefig(run_dir / "generated_images.png", dpi=150)
    plt.close(fig)

    comparison_steps = 4 if 4 in generated else sampling_steps[-1]
    comparison = generated[comparison_steps]
    n_parameters_plot = min(3, validation_parameters.shape[1])
    fig, axes = plt.subplots(1, n_parameters_plot + 1, figsize=(4 * (n_parameters_plot + 1), 3), layout="constrained")
    axes[0].bar(count_values, count_probability, alpha=0.5, label="Train count prior")
    axes[0].hist(generated_counts, bins=np.arange(capacity + 2) - 0.5, density=True,
                 histtype="step", label="Generated")
    axes[0].set(xlabel="Galaxy count", ylabel="Probability", title="Discrete count model")
    axes[0].legend()
    for j in range(n_parameters_plot):
        axes[j + 1].hist(validation_parameters[:, j], bins=12, density=True, histtype="step", label="Held out")
        axes[j + 1].hist(comparison["parameters"][:, j], bins=12, density=True, histtype="step", label="Generated")
        axes[j + 1].set(title=str(parameter_names[j]), ylabel="Density")
        axes[j + 1].legend()
    fig.savefig(run_dir / "generated_marginals.png", dpi=150)
    plt.close(fig)


def run(args):
    """Prepare data, train the conditional flow map, checkpoint and generate samples."""
    if args.data is None:
        archives = sorted(args.data_dir.glob("*/data.npz"), key=lambda path: path.parent.name)
        if not archives:
            raise FileNotFoundError(f"No */data.npz under {args.data_dir.resolve()}; run galsbi_dataset.py first or pass --data.")
        data_path = archives[-1]
    else:
        data_path = args.data
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {data_path.resolve()}")
    print("Dataset:", data_path.resolve())
    print("JAX devices:", jax.devices())

    required = ("images", "bands", "parameters", "parameter_names", "model_indices",
                "truth_catalog", "truth_mask", "truth_counts", "truth_feature_names")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in required}
    images = np.asarray(data["images"], dtype=np.float32)
    catalog = np.asarray(data["truth_catalog"], dtype=np.float32)
    catalog_mask = np.asarray(data["truth_mask"], dtype=bool)
    counts = np.asarray(data["truth_counts"], dtype=np.int64)
    bands = data["bands"].astype(str).tolist()
    feature_names = data["truth_feature_names"].astype(str).tolist()
    all_parameter_names = data["parameter_names"].astype(str)
    physical_prefixes = ("ell_beta_", "logr50_", "lum_fct_", "sersic_", "template_coeff_")
    parameter_selection = np.array([name.startswith(physical_prefixes) for name in all_parameter_names])
    parameter_names = all_parameter_names[parameter_selection]
    parameters = np.asarray(data["parameters"][:, parameter_selection], dtype=np.float32)
    model_indices = np.asarray(data["model_indices"])

    n_data, n_channels, height, width = images.shape
    assert n_data >= 2 and len(bands) == n_channels
    assert catalog.ndim == 3 and catalog.shape[:2] == catalog_mask.shape
    assert len(catalog) == len(parameters) == len(counts) == len(model_indices) == n_data
    assert len(feature_names) == catalog.shape[-1] and len(set(feature_names)) == len(feature_names)
    assert parameters.shape[1] > 0 and np.isfinite(parameters).all() and np.isfinite(images).all()
    assert np.array_equal(catalog_mask.sum(axis=1), counts) and np.all(counts >= 0)
    assert np.isfinite(catalog[catalog_mask]).all()
    expected_features = {"x", "y", "z", "r50", "sersic_n", "e1", "e2", "galaxy_type"} | {f"mag_{b}" for b in bands}
    assert set(feature_names) == expected_features, "This example expects the 13-feature intrinsic catalog schema."
    feature = {name: feature_names.index(name) for name in feature_names}
    assert np.isin(catalog[..., feature["galaxy_type"]][catalog_mask], [0.0, 1.0]).all()
    # A zero-object archive still needs a static token capacity of at least one.
    capacity = max(1, catalog.shape[1])
    if catalog.shape[1] == 0:
        catalog = np.zeros((n_data, capacity, len(feature_names)), dtype=np.float32)
        catalog_mask = np.zeros((n_data, capacity), dtype=bool)

    # Keep repeated posterior rows together. A single-row archive splits realizations.
    groups = np.unique(model_indices)
    grouped = len(groups) > 1
    units = groups if grouped else np.arange(n_data)
    shuffled = np.random.default_rng(args.seed).permutation(units)
    n_val = min(len(units) - 1, max(1, round(args.validation_fraction * len(units))))
    is_validation = np.isin(model_indices if grouped else units, shuffled[:n_val])
    train_indices, val_indices = np.flatnonzero(~is_validation), np.flatnonzero(is_validation)
    split_kind = "grouped by model_indices" if grouped else "realization split: only one population-parameter row available"
    print(split_kind, "| train:", len(train_indices), "validation:", len(val_indices))

    # Fit only training images/parameters and valid training galaxies.
    stats = {"image_softening": np.maximum(images[train_indices].std(axis=(0, 2, 3), keepdims=True), 1e-3)}
    image_transformed = np.arcsinh(images / stats["image_softening"])
    normalized_images = standardize(image_transformed, image_transformed[train_indices], "image", stats, axis=(0, 2, 3), keepdims=True)
    normalized_parameters = standardize(parameters, parameters[train_indices], "parameter", stats)
    catalog_transformed = catalog_coordinates(catalog, catalog_mask, feature, width, height)
    training_rows = catalog_transformed[train_indices][catalog_mask[train_indices]]
    normalized_catalog = standardize(catalog_transformed, training_rows, "catalog", stats)
    normalized_catalog = np.where(catalog_mask[..., None], normalized_catalog, 0.0)
    count_values = np.arange(capacity + 1, dtype=np.int64)
    count_probability = np.bincount(counts[train_indices], minlength=capacity + 1).astype(np.float64)
    count_probability /= count_probability.sum()
    stats["count_mean"] = np.float32(np.log1p(counts[train_indices]).mean())
    stats["count_scale"] = safe_scale(np.log1p(counts[train_indices]).std())
    normalized_counts = ((np.log1p(counts) - stats["count_mean"]) / stats["count_scale"]).astype(np.float32)[:, None]
    print("Training count support:", count_values[count_probability > 0])
    print("Valid training galaxies:", len(training_rows))

    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "config.json"
    config = set_model_family(get_config(config_path, reload=False, seed=args.seed), "flow_map")
    config.model_config.backbone.update(
        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, n_heads=args.heads,
        patch_size=args.patch_size, dropout_rate=0.0,
    )
    config.model_config.family.update(n_blocks=args.blocks, t_emb_dim=32)
    config.train_config.update(
        n_steps=args.steps, n_batch=args.batch_size, flow_map_loss=args.loss,
        flow_map_psd_split="uniform", flow_map_diagonal_weight=1.0,
        flow_map_distillation_weight=1.0, conditional_training_probability=0.0,
        clip_grad_norm=args.grad_clip, ema_rate=args.ema_rate, use_ema=True,
        opt_kwargs={"learning_rate": args.learning_rate},
    )
    config.sampling_config.flow_map_steps = 4
    save_json_config(config_path, config)

    modality_infos = ModalityInfos({
        "image": ModalitySpec("image", "image", tuple(map(int, images.shape[1:])), patch_size=args.patch_size, embedding_dim=args.embedding_dim),
        "catalog": ModalitySpec("catalog", "set", tuple(map(int, catalog.shape[1:])), embedding_dim=args.embedding_dim),
        "parameters": ModalitySpec("parameters", "vector", (int(parameters.shape[1]),), embedding_dim=args.embedding_dim),
        "count": ModalitySpec("count", "vector", (1,), embedding_dim=args.embedding_dim),
    })
    family = build_model_family(config, modality_infos, jr.key(args.seed))
    model, ema_model = family.model, family.model
    conditioning_mask = jnp.array([name == "count" for name in sorted(modality_infos)])
    optimiser = optax.chain(optax.clip_by_global_norm(args.grad_clip), optax.adam(args.learning_rate))
    opt_state = optimiser.init(eqx.filter(model, eqx.is_inexact_array))
    print("Parameters:", count_parameters(model), "| token lengths:", model.get_modality_lengths())
    print("Run:", run_dir.resolve())

    # Fresh type dequantization and valid-row shuffling happen on the host.
    def make_batch(indices, rng):
        indices = np.asarray(indices)
        mask = catalog_mask[indices].copy()
        rows = normalized_catalog[indices].copy()
        raw_type = np.where(mask, catalog[indices, :, feature["galaxy_type"]], 0.0)
        uniform = rng.uniform(EPS, 1.0 - EPS, size=raw_type.shape)
        rows[..., feature["galaxy_type"]] = np.where(mask, logit((raw_type + uniform) / 2.0), 0.0)
        for i in range(len(indices)):
            valid_rows = np.flatnonzero(mask[i])
            rows[i, valid_rows] = rows[i, rng.permutation(valid_rows)]
        batch = {
            "image": jnp.asarray(normalized_images[indices]),
            "catalog": jnp.asarray(rows),
            "parameters": jnp.asarray(normalized_parameters[indices]),
            "count": jnp.asarray(normalized_counts[indices]),
        }
        return batch, {"catalog": jnp.asarray(mask)}

    @eqx.filter_jit
    def train_step(model, ema_model, opt_state, batch, masks, key):
        (loss, metrics), gradients = eqx.filter_value_and_grad(family.loss_fn, has_aux=True)(
            model, batch, key, masks=masks, conditioning_mask=conditioning_mask,
        )
        updates, opt_state = optimiser.update(gradients, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        ema_model = apply_ema(ema_model, model, ema_rate=args.ema_rate)
        return model, ema_model, opt_state, loss, metrics

    @eqx.filter_jit
    def validation_loss(model, batch, masks):
        return family.loss_fn(
            eqx.nn.inference_mode(model), batch, jr.key(args.seed + 4),
            masks=masks, conditioning_mask=conditioning_mask,
        )[0]

    batch_rng = np.random.default_rng(args.seed + 1)
    validation_indices = val_indices[:min(args.batch_size, len(val_indices))]
    val_batch, val_masks = make_batch(validation_indices, np.random.default_rng(args.seed + 3))
    training_key = jr.key(args.seed + 2)
    history = []
    for step in range(1, args.steps + 1):
        indices = batch_rng.choice(train_indices, size=args.batch_size, replace=True)
        batch, masks = make_batch(indices, batch_rng)
        training_key, loss_key = jr.split(training_key)
        model, ema_model, opt_state, loss, metrics = train_step(
            model, ema_model, opt_state, batch, masks, loss_key,
        )
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {"step": step, "train": float(loss),
                      "validation": float(validation_loss(ema_model, val_batch, val_masks)),
                      "flow_matching": float(metrics["flow_matching_loss"]),
                      "distillation": float(metrics["distillation_loss"]),
                      **{f"loss_{name}": float(value) for name, value in metrics["per_modality_loss"].items()}}
            history.append(record)
            print(record)
    assert history, "Training must include at least one update."

    plot_history(history, args.loss, run_dir)

    checkpoint_path = run_dir / "checkpoint.eqx"
    normalization_path = run_dir / "normalization.npz"
    save_training_state(checkpoint_path, model, ema_model, optimiser, opt_state, args.steps)
    np.savez_compressed(
        normalization_path,
        **stats, count_values=count_values,
        count_probability=count_probability, parameter_names=parameter_names,
        feature_names=np.asarray(feature_names), bands=np.asarray(bands),
        image_shape=np.asarray(images.shape[1:]), catalog_shape=np.asarray(catalog.shape[1:]),
        train_indices=train_indices, val_indices=val_indices, validation_indices=validation_indices,
        parameter_selection=parameter_selection, training_key=np.asarray(jr.key_data(training_key)),
    )
    metadata = {
        "data_path": str(data_path.resolve()), "split": split_kind, "seed": args.seed,
        "loss_type": args.loss, "steps": args.steps, "batch_size": args.batch_size,
        "normalization_epsilon": EPS, "ema_rate": args.ema_rate,
        "versions": {name: version(name) for name in ("joint", "jax", "equinox", "optax", "numpy")},
        "factorization": "empirical P(N) times joint flow map P(image,catalog,parameters|N)",
        "catalog_type_transform": "fresh u~Uniform(eps,1-eps); logit((galaxy_type+u)/2); decode logit>=0",
        "catalog_ellipticity_transform": "q=e/(1-norm(e)); inverse e=q/(1+norm(q))",
        "catalog_position_transform": "logit(x/width), logit(y/height), clipped only at epsilon endpoints",
        "catalog_positive_transform": "log(max(value,eps)) for z,r50,sersic_n",
        "batch_rng_state": batch_rng.bit_generator.state,
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Saved", checkpoint_path)

    # Reload into the existing structure to check the saved checkpoint.
    model, ema_model, optimiser, opt_state, restored_step = load_training_state(checkpoint_path, model, optimiser, opt_state)
    assert restored_step == args.steps
    print("Reloaded completed step:", restored_step)

    # All three generated modalities start with independent noise; count stays fixed.
    sampling_rng = np.random.default_rng(args.seed + 10)
    generated_counts = sampling_rng.choice(count_values, size=args.n_generate, p=count_probability)
    generated_mask = np.arange(capacity)[None, :] < generated_counts[:, None]
    image_key, catalog_key, parameter_key = jr.split(jr.key(args.seed + 11), 3)
    base_state = {
        "image": jr.normal(image_key, (args.n_generate,) + images.shape[1:]),
        "catalog": jnp.where(jnp.asarray(generated_mask)[..., None],
                              jr.normal(catalog_key, (args.n_generate,) + catalog.shape[1:]), 0.0),
        "parameters": jr.normal(parameter_key, (args.n_generate, parameters.shape[1])),
        "count": jnp.asarray(((np.log1p(generated_counts) - stats["count_mean"]) / stats["count_scale"]).astype(np.float32)[:, None]),
    }
    generated = {}
    for n_steps in args.sampling_steps:
        samples = jax.vmap(lambda state, mask: single_sample_modalities_fn(
            ema_model, state, ["catalog", "image", "parameters"],
            masks={"catalog": mask}, n_steps=n_steps,
        ))(base_state, jnp.asarray(generated_mask))
        samples = jax.tree.map(np.asarray, samples)
        assert all(np.isfinite(value).all() for value in samples.values())
        np.testing.assert_array_equal(samples["count"], np.asarray(base_state["count"]))
        np.testing.assert_array_equal(samples["catalog"][~generated_mask], 0.0)
        generated[n_steps] = decode_samples(samples, generated_mask, stats, feature, width, height)
        np.savez_compressed(
            run_dir / f"generated_{n_steps:02d}_steps.npz", **generated[n_steps],
            catalog_mask=generated_mask, catalog_counts=generated_counts,
            feature_names=np.asarray(feature_names), parameter_names=parameter_names,
            bands=np.asarray(bands), sampling_steps=np.asarray(n_steps),
        )
    print("Drawn counts:", generated_counts)

    plot_samples(generated, generated_mask, generated_counts, bands, feature,
                 args.sampling_steps, parameters[val_indices], parameter_names, count_probability, run_dir)
    print("Finished:", run_dir.resolve())
    return run_dir


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
