#!/usr/bin/env python3
"""Generate paired grizy images, intrinsic galaxy sets and population parameters.

Run with ``uv run --group examples python scripts/galsbi_dataset.py --help``.
Posterior rows are sampled with replacement; spaced seeds give independent skies.
Empty catalogs are retained, with masks distinguishing real galaxies from padding.
Optional SExtractor measurements remain separate from attached simulation truth.

Paths are relative to the working directory. Arrays accumulate in memory, so split
large runs into batches. Group training/validation by model_indices to assess new
parameter settings, and fit normalization on training data only.
Documentation: https://cosmo-docs.phys.ethz.ch/galsbi/
"""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import json
import shutil

import numpy as np

BANDS = ("g", "r", "i", "z", "y")
MODEL_NAME = "Fischbacher+24"
NSIDE_SAMPLING = 8192  # Reduce discarded sky area without changing galaxy density.
SEED_STRIDE = 10_000  # Separate realizations beyond UFig's per-band RNG offsets.
PIXEL_SCALE = 0.168  # arcsec/pixel; a 32-pixel patch spans only 5.376 arcsec.
TRUTH_COLUMNS = ("x", "y", "z", "r50", "sersic_n", "e1", "e2", "galaxy_type")
MEASURED_COLUMNS = ("X_IMAGE", "Y_IMAGE", "FLUX_AUTO", "FLUXERR_AUTO",
                    "MAG_AUTO", "MAGERR_AUTO", "FLUX_RADIUS", "FLAGS")
TRUTH_NAMES = list(TRUTH_COLUMNS) + [f"mag_{band}" for band in BANDS]
OBSERVED_NAMES = [f"{column}_{band}" for band in BANDS for column in MEASURED_COLUMNS]
TRUTH_UNITS = ["pixel (edge origin)", "pixel (edge origin)", "redshift", "pixel",
               "dimensionless", "dimensionless", "dimensionless", "label"] + ["mag"] * len(BANDS)
OBSERVED_UNITS = ["pixel (center origin 1)", "pixel (center origin 1)", "ADU", "ADU",
                  "mag", "mag", "pixel", "bit flag"] * len(BANDS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--n-samples", type=int, default=100, help="Number of sky realizations.")
    parser.add_argument("--n-pix", type=int, default=64, help="Square image size in pixels.")
    parser.add_argument("--output-dir", type=Path, default=Path("out/galsbi"), help="Parent output directory.")
    parser.add_argument("--seed", type=int, default=20261108, help="Base simulation seed; stride 10000.")
    parser.add_argument("--parameter-seed", type=int, default=42, help="Seed for posterior-row selection.")
    parser.add_argument("--source-extraction", action="store_true", help="Also run external SExtractor.")
    args = parser.parse_args(argv)
    if args.n_samples < 1 or args.n_pix < 1 or min(args.seed, args.parameter_seed) < 0:
        parser.error("sizes must be positive and seeds nonnegative")
    if args.seed + SEED_STRIDE * (args.n_samples - 1) + 12 * NSIDE_SAMPLING**2 >= 2**32:
        parser.error("simulation seeds plus HEALPix indices exceed the supported range")
    return args


def align_bands(catalogs, prefix, id_column):
    """Sort unique source IDs and require the same ID set in every band."""
    aligned, reference_ids = {}, None
    for band in BANDS:
        catalog = catalogs[f"{prefix} {band}"]
        ids = np.asarray(catalog[id_column])
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Duplicate {id_column} in {prefix} {band}")
        order = np.argsort(ids)
        ids = ids[order]
        if reference_ids is None:
            reference_ids = ids
        elif not np.array_equal(ids, reference_ids):
            raise ValueError(f"Different {id_column} sets in {prefix} {band}; cannot align bands")
        aligned[band] = catalog[order]
    return aligned, reference_ids.astype(np.int64)


def load_sample(model, index, n_pix, source_extraction):
    images = model.load_images(model_index=int(index))
    catalogs = model.load_catalogs(model_index=int(index), combine=False)
    image = np.stack([images[f"image {band}"] for band in BANDS]).astype(np.float32)
    assert image.shape == (len(BANDS), n_pix, n_pix) and np.isfinite(image).all()
    truth, ids = align_bands(catalogs, "ucat galaxies", "id")
    sample = {"image": image, "truth_ids": ids, "truth": np.column_stack(
        [truth["i"][column] for column in TRUTH_COLUMNS] + [truth[band]["mag"] for band in BANDS]
    ).astype(np.float32)}
    if source_extraction:
        observed, numbers = align_bands(catalogs, "sextractor", "NUMBER")
        sample["observed_ids"] = numbers
        sample["observed"] = np.column_stack([
            observed[band][column] for band in BANDS for column in MEASURED_COLUMNS
        ]).astype(np.float32)
        # Truth matches are metadata, never measured features; -200 means unmatched.
        sample["matched_ids"] = np.asarray(observed["i"]["id"], dtype=np.int64)
        assert np.all((sample["matched_ids"] == -200) | np.isin(sample["matched_ids"], ids))
    return sample


def pack_catalog(samples, prefix, names, units):
    """Pad a collection of sets without truncation, including entirely empty sets."""
    counts = np.array([len(sample[prefix]) for sample in samples], dtype=np.int64)
    shape = (len(samples), int(counts.max(initial=0)))
    values = np.zeros(shape + (len(names),), dtype=np.float32)
    mask = np.zeros(shape, dtype=bool)
    ids = np.full(shape, -1, dtype=np.int64)
    for i, sample in enumerate(samples):
        values[i, :counts[i]] = sample[prefix]
        mask[i, :counts[i]] = True
        ids[i, :counts[i]] = sample[f"{prefix}_ids"]
    return {f"{prefix}_catalog": values, f"{prefix}_mask": mask, f"{prefix}_counts": counts,
            f"{prefix}_ids": ids, f"{prefix}_feature_names": np.asarray(names),
            f"{prefix}_feature_units": np.asarray(units)}


@contextmanager
def allow_empty_psf_catalogs():
    """Guard UFig 1.2.1's mean(empty) bug; keep nonempty PSF calculations unchanged."""
    from ufig.plugins import add_psf
    original = add_psf.multiple_moffat_r50

    def empty_safe_radius(fwhm, *args, **kwargs):
        return np.empty_like(fwhm) if np.size(fwhm) == 0 else original(fwhm, *args, **kwargs)

    add_psf.multiple_moffat_r50 = empty_safe_radius
    try:
        yield
    finally:
        add_psf.multiple_moffat_r50 = original


def json_value(value):
    """Snapshot simulator metadata as plain JSON values before the next run."""
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def save_diagnostic_plot(run_dir, sample, n_pix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image, truth = sample["image"][BANDS.index("i")], sample["truth"]
    low, high = np.percentile(image, [1, 99.5])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].imshow(image, origin="lower", extent=(0, n_pix, 0, n_pix), cmap="gray", vmin=low, vmax=high)
    axes[0].set_title("i-band image")
    points = axes[1].scatter(truth[:, 0], truth[:, 1], c=truth[:, TRUTH_NAMES.index("z")],
                             s=14, cmap="viridis", label="Intrinsic galaxies")
    axes[1].set_title("Paired intrinsic source set")
    if "observed" in sample:
        observed = sample["observed"]
        # SExtractor's first pixel center is 1; UFig's is 0.5.
        x = observed[:, OBSERVED_NAMES.index("X_IMAGE_i")] - 0.5
        y = observed[:, OBSERVED_NAMES.index("Y_IMAGE_i")] - 0.5
        for axis in axes:
            axis.scatter(x, y, marker="+", s=45, c="tab:orange", label="SExtractor detections")
            axis.legend(loc="upper right")
    fig.colorbar(points, ax=axes[1], label="True redshift")
    for axis in axes:
        axis.set(xlim=(0, n_pix), ylim=(0, n_pix), xlabel="x [pixel, edge origin]",
                 ylabel="y [pixel, edge origin]", aspect="equal")
    fig.savefig(run_dir / "dataset_preview.png", dpi=150)
    plt.close(fig)


def run_generation(args):
    sextractor = shutil.which("sex") or shutil.which("source-extractor")
    if args.source_extraction and sextractor is None:
        raise RuntimeError("Install SExtractor and put 'sex' or 'source-extractor' on PATH.")
    from galsbi import GalSBI
    from galsbi.load import load_abc_posterior

    mode = "image+SE" if args.source_extraction else "image"
    seeds = args.seed + SEED_STRIDE * np.arange(args.n_samples, dtype=np.int64)
    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ")
    raw_dir, log_path = run_dir / "raw", run_dir / "simulation.log"
    raw_dir.mkdir(parents=True, exist_ok=False)
    print("Output:", run_dir.resolve(), "\nSimulator log:", log_path.resolve(), flush=True)

    posterior = load_abc_posterior(MODEL_NAME)
    if len(posterior) == 0:
        raise ValueError("The GalSBI posterior contains no parameter rows.")
    model_indices = np.random.default_rng(args.parameter_seed).integers(
        len(posterior), size=args.n_samples, dtype=np.int64
    )
    fields = [name for name in posterior.dtype.names
              if name.startswith(("ell_", "logr50_", "lum_fct_", "sersic_", "template_coeff_"))]
    if not fields:
        raise ValueError("Select physical parameter fields for the chosen galaxy model.")
    names, layout, blocks = [], {}, []
    for name in fields:
        values = np.asarray(posterior[name][model_indices])
        if values.dtype.kind not in "biuf":
            raise TypeError(f"Expected numeric posterior parameter {name!r}, got {values.dtype}")
        shape = values.shape[1:]
        layout[name] = list(shape)
        names.extend(name if not shape else f"{name}[{','.join(map(str, index))}]"
                     for index in np.ndindex(shape))
        blocks.append(values.reshape(args.n_samples, -1).astype(np.float64))
    parameters = np.concatenate(blocks, axis=1)
    assert np.isfinite(parameters).all()
    print(f"Parameter matrix: {parameters.shape}; using {len(np.unique(model_indices))} posterior rows")

    nuisance_names = (
        "size_x", "size_y", "pixscale", "ra0", "dec0", "seed", "reference_band", "nside_sampling",
        "seeing", "psf_type", "psf_beta", "psf_e1", "psf_e2", "background_type",
        "background_sigma_dict", "bkg_noise_amp_dict", "gain_dict", "n_exp_dict",
        "magzero_dict", "gals_mag_min", "gals_mag_max",
    )
    if args.source_extraction:
        nuisance_names += ("sextractor_forced_photo_detection_bands", "sextractor_config", "sextractor_params")
    samples = []
    for i, (index, seed) in enumerate(zip(model_indices, seeds, strict=True)):
        prefix = raw_dir / f"sample_{i:04d}_seed_{int(seed)}"
        model = GalSBI(MODEL_NAME, verbosity="warning")
        with log_path.open("a", encoding="utf-8") as log, allow_empty_psf_catalogs(), redirect_stdout(log), redirect_stderr(log):
            model(mode=mode, model_index=int(index), seed=int(seed), file_name=str(prefix),
                  size_x=args.n_pix, size_y=args.n_pix, pixscale=PIXEL_SCALE, ra0=0.0, dec0=0.0,
                  filters=list(BANDS), nside_sampling=NSIDE_SAMPLING,
                  sextractor_forced_photo_detection_bands=["i"],
                  **({"sextractor_binary": sextractor} if args.source_extraction else {}))
        sample = load_sample(model, index, args.n_pix, args.source_extraction)
        sample["metadata"] = {
            "sample": i, "model_index": int(index), "seed": int(seed),
            "raw_prefix": str(prefix.relative_to(run_dir)),
            "nuisance": {name: json_value(getattr(model.ctx.parameters, name)) for name in nuisance_names},
        }
        samples.append(sample)
        if i == 0 or (i + 1) % max(1, args.n_samples // 100) == 0 or i + 1 == args.n_samples:
            summary = f"{i + 1}/{args.n_samples}: theta row {index}, seed {seed}, {len(sample['truth'])} intrinsic galaxies"
            if args.source_extraction:
                summary += f", {len(sample['observed'])} detected sources"
            print(summary, flush=True)

    arrays = {"images": np.stack([sample["image"] for sample in samples]), "bands": np.asarray(BANDS),
              "parameters": parameters, "parameter_names": np.asarray(names),
              "model_indices": model_indices, "seeds": seeds}
    arrays.update(pack_catalog(samples, "truth", TRUTH_NAMES, TRUTH_UNITS))
    if args.source_extraction:
        arrays.update(pack_catalog(samples, "observed", OBSERVED_NAMES, OBSERVED_UNITS))
        values, mask = arrays["observed_catalog"], arrays["observed_mask"]
        matched_ids = np.full(mask.shape, -1, dtype=np.int64)
        matched_mask = np.zeros_like(mask)
        for i, sample in enumerate(samples):
            ids = sample["matched_ids"]
            matched_ids[i, :len(ids)] = ids
            matched_mask[i, :len(ids)] = np.isin(ids, sample["truth_ids"])
        valid = mask[..., None] & np.isfinite(values)
        for j, name in enumerate(OBSERVED_NAMES):
            if name.startswith(("MAG_AUTO_", "MAGERR_AUTO_")):
                valid[..., j] &= values[..., j] < 90.0
        arrays.update(observed_feature_valid=valid, observed_truth_ids=matched_ids,
                      observed_has_truth_match=matched_mask)
    assert all(not value.dtype.hasobject for value in arrays.values())
    np.savez_compressed(run_dir / "data.npz", **arrays)
    manifest = {
        "format_version": 2, "model": MODEL_NAME, "mode": mode, "bands": list(BANDS),
        "versions": {name: version(name) for name in ("galsbi", "ufig", "numpy")},
        "sextractor_binary": sextractor if args.source_extraction else None,
        "parameter_field_shapes": layout, "parameter_fields": fields,
        "parameter_selection": "uniform posterior rows with replacement",
        "parameter_selection_seed": args.parameter_seed, "posterior_rows": len(posterior),
        "simulation_seed_base": args.seed, "simulation_seed_stride": SEED_STRIDE,
        "image_layout": "sample, band, y, x", "image_units": "ADU pixel counts",
        "truth_coordinates": "UFig pixels: edge 0, first pixel center 0.5",
        "observed_coordinates": "SExtractor pixels: first pixel center 1; subtract 0.5 for UFig coordinates",
        "catalog_id_scope": "IDs are local to each realization; truth id and observed NUMBER are different namespaces",
        "catalog_alignment": "sorted unique truth id / observed NUMBER; identical ID sets verified across bands",
        "padding": "zero values, ID -1, masks false; no truncation", "observed_truth_unmatched_id": -200,
        "stars": "disabled in the stock Fischbacher+24 image+SE plugin list",
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in arrays.items()},
        "realizations": [sample["metadata"] for sample in samples],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    save_diagnostic_plot(run_dir, samples[0], args.n_pix)
    print("Saved data.npz, manifest.json and dataset_preview.png in", run_dir)
    return run_dir


def main(argv=None):
    run_generation(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
