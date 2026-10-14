from pathlib import Path
import importlib
import json
import  diffrax as dfx
import optax

from .generative.flow_matching import OptimalTransportPath


MODEL_BACKBONE_DEFAULTS = {
    "patch_size": 2,
    "hidden_dim": 256,
    "n_heads": 8,
    "embedding_dim": 32,
    "dropout_rate": 0.0,
}


MODEL_FAMILY_DEFAULTS = {
    "flow_matching": {
        "n_blocks": 4,
        "t_emb_dim": 32,
        "sigma_min": 1e-3,
    },
    "sphere_encoder": {
        "n_blocks_encoder": 8,
        "n_blocks_decoder": 8,
        "latent_dim": 32,
        "noise_sigma_max_angle": 75.0,
    },
}


MODEL_FAMILY_CHOICES = tuple(sorted(MODEL_FAMILY_DEFAULTS))


class ConfigNode(dict):
    """Dictionary with attribute-style access.

    Inputs:
        name (str): Attribute name requested by callers.
        value (Any): Value assigned by callers.

    Outputs:
        Any: Stored configuration value for attribute/key lookups.
    """
    __slots__ = ()

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name, value):
        self[name] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, _to_config_node(value))


def _to_config_node(value):
    """Recursively convert nested containers to ``ConfigNode``.

    Inputs:
        value (Any): Python object that may contain nested dict/list/tuple values.

    Outputs:
        Any: Converted object where dictionaries become ``ConfigNode`` instances.
    """
    if isinstance(value, ConfigNode):
        return value
    if isinstance(value, dict):
        node = ConfigNode()
        for key, item in value.items():
            node[key] = item
        return node
    if isinstance(value, list):
        return [_to_config_node(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_config_node(item) for item in value)
    return value


def _build_default_model_config(model_family: str = "flow_matching"):
    """Build the default model config for a given family."""
    if model_family not in MODEL_FAMILY_DEFAULTS:
        raise ValueError(
            f"Unknown model_family {model_family!r}. Expected one of {sorted(MODEL_FAMILY_DEFAULTS)}."
        )
    return {
        "model_family": model_family,
        "backbone": dict(MODEL_BACKBONE_DEFAULTS),
        "family": dict(MODEL_FAMILY_DEFAULTS.get(model_family, {})),
    }


def set_model_family(config, model_family: str):
    """Update a config to use a specific model family.

    Keeps the current shared backbone settings. If the requested family matches
    the current family, existing family-specific overrides are preserved on top
    of that family's defaults. If the family changes, the family block is reset
    to the new family's defaults.
    """
    model_config = _build_default_model_config(model_family)
    current_model_config = getattr(config, "model_config", None)
    if isinstance(current_model_config, dict):
        current_backbone = current_model_config.get("backbone")
        if isinstance(current_backbone, dict):
            shared_backbone = {
                key: current_backbone[key]
                for key in MODEL_BACKBONE_DEFAULTS
                if key in current_backbone
            }
            model_config["backbone"].update(shared_backbone)

        current_family_name = current_model_config.get("model_family")
        current_family = current_model_config.get("family")
        if (
            current_family_name == model_family
            and isinstance(current_family, dict)
        ):
            model_config["family"].update(current_family)

    config.model_config = model_config
    return config


def _validate_config_schema(config):
    """Validate that a config uses the canonical nested schema.

    Inputs:
        config (dict): Configuration dictionary loaded from disk.

    Outputs:
        dict: The validated configuration dictionary.
    """
    if not isinstance(config, dict):
        raise TypeError("Config must deserialize to a dictionary.")

    required_sections = ("data_config", "model_config", "train_config", "sampling_config")
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(
            "Config is missing required top-level sections: "
            + ", ".join(missing_sections)
        )

    model_config = config["model_config"]
    if not isinstance(model_config, dict):
        raise TypeError("model_config must be a dictionary.")

    required_model_keys = ("model_family", "backbone", "family")
    missing_model_keys = [
        section for section in required_model_keys if section not in model_config
    ]
    if missing_model_keys:
        raise ValueError(
            "model_config must define: " + ", ".join(required_model_keys)
        )

    model_family = model_config["model_family"]
    if not isinstance(model_family, str):
        raise TypeError("model_config.model_family must be a string.")
    if model_family not in MODEL_FAMILY_DEFAULTS:
        raise ValueError(
            f"Unknown model_family {model_family!r}. Expected one of {sorted(MODEL_FAMILY_DEFAULTS)}."
        )
    if not isinstance(model_config["backbone"], dict):
        raise TypeError("model_config.backbone must be a dictionary.")
    if not isinstance(model_config["family"], dict):
        raise TypeError("model_config.family must be a dictionary.")

    return config


def _merge_missing_defaults(config, defaults):
    """Recursively fill missing config entries from defaults."""
    if isinstance(defaults, ConfigNode):
        defaults = dict(defaults)
    if isinstance(config, ConfigNode):
        config = dict(config)
    if isinstance(defaults, dict) and isinstance(config, dict):
        merged = dict(config)
        for key, default_value in defaults.items():
            if key not in merged:
                merged[key] = default_value
            else:
                merged[key] = _merge_missing_defaults(merged[key], default_value)
        return _to_config_node(merged)
    return config


def _serialize_config(value):
    """Serialize config values into a JSON-safe structure.

    Inputs:
        value (Any): Config value that may include callables, paths, tuples, or dicts.

    Outputs:
        Any: JSON-serializable representation including marker dicts for special types.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_serialize_config(item) for item in value]
    if isinstance(value, tuple):
        return {"__kind__": "tuple", "items": [_serialize_config(item) for item in value]}
    if isinstance(value, dict):
        return {str(key): _serialize_config(item) for key, item in value.items()}
    if isinstance(value, Path):
        return {"__kind__": "path", "value": str(value)}
    if callable(value):
        return {
            "__kind__": "python_ref",
            "module": value.__module__,
            "qualname": value.__qualname__,
        }
    raise TypeError(f"Unsupported config type: {type(value)!r}")


def _resolve_python_ref(module, qualname):
    """Resolve a dotted Python reference.

    Inputs:
        module (str): Importable module path.
        qualname (str): Dotted attribute path inside the module.

    Outputs:
        Any: Resolved Python object.
    """
    obj = importlib.import_module(module)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _deserialize_config(value):
    """Deserialize config values saved by ``_serialize_config``.

    Inputs:
        value (Any): Raw value decoded from JSON.

    Outputs:
        Any: Reconstructed Python value with paths/callables/tuples restored.
    """
    if isinstance(value, list):
        return [_deserialize_config(item) for item in value]
    if not isinstance(value, dict):
        return value

    kind = value.get("__kind__")
    if kind == "tuple":
        return tuple(_deserialize_config(item) for item in value["items"])
    if kind == "path":
        return Path(value["value"])
    if kind == "python_ref":
        return _resolve_python_ref(value["module"], value["qualname"])
    if kind is None:
        return {key: _deserialize_config(item) for key, item in value.items()}
    raise ValueError(f"Unknown config marker: {kind!r}")


def save_json_config(path, config):
    """Write configuration to disk as compact JSON.

    Inputs:
        path (Path | str): Destination file path.
        config (dict | ConfigNode): Configuration tree to serialize.

    Outputs:
        None: Writes the serialized configuration file.
    """
    encoded = _serialize_config(config)
    path.write_text(json.dumps(encoded, separators=(",", ":")), encoding="utf-8")


def load_json_config(path):
    """Load configuration JSON from disk.

    Inputs:
        path (Path | str): Path to a serialized configuration file.

    Outputs:
        ConfigNode: Loaded and schema-validated configuration.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    decoded = _deserialize_config(raw)
    return _to_config_node(_validate_config_schema(decoded))


def get_config(
    config_path: Path | str, 
    *, 
    reload: bool = True, 
    seed: int = 0
) -> ConfigNode:
    """Load an existing config or create a default one.

    Inputs:
        config_path (Path | str): Config file location.
        reload (bool): If ``True`` and the file exists, load it from disk.
        seed (int): Default random seed used when creating a new config.

    Outputs:
        ConfigNode: Active configuration used by training/inference scripts.
    """
    base_default_config = {
        "seed": seed,
        "data_config": {
            "n_samples": 10_000,
            "n_channels": 1,
            "n_pix": 32,
            "parameter_shape": (2,),
            "val_fraction": 0.1,
        },
        "train_config": {
            "n_steps": 10_000_000,
            "n_batch": 256,
            "n_batch_val": 256,
            "opt_factory": optax.adamw,
            "opt_kwargs": {"learning_rate": 2e-4},
            "clip_grad_norm": 1.0,
            "use_lr_schedule": True,
            "warmup_steps": 1_000,
            "lr_end_value": 2e-5,
            "use_ema": True,
            "sample_every": 500,
            "n_steps_validate": 100,
            "time_sampling_distribution": "uniform",
            "time_sampling_beta_alpha": 1.0,
            "time_sampling_beta_beta": 1.0,
            "time_sampling_stratified": True,
            "n_t_bins": 4,
            "conditional_training_probability": 1.0,
            "reconstruction_loss": "l1+l2",
            "reconstruction_perceptual_weight": 0.0,
            "pixel_consistency_weight": 0.25,
            "pixel_consistency_perceptual_weight": 0.0,
            "latent_consistency_weight": 0.10,
            "mix_hard_cases": True,
            "mix_hard_cases_prob": 0.10,
            "mix_hard_cases_max_angle": 80.0,
        },
        "sampling_config": {
            "n_samples": 4,
            "t0": 0.0,
            "t1": 1.0,
            "dt0": 1e-2,
            "solver_factory": dfx.Heun,
            "path_factory": OptimalTransportPath,
            "forward_steps": 1,
            "use_sampling_scheduler": False,
            "cache_sampling_noise": False,
        },
    }

    if reload and config_path.exists():
        loaded_config = load_json_config(config_path)
        model_family = str(loaded_config.model_config.model_family)
        default_config = _to_config_node(
            {
                **base_default_config,
                "model_config": _build_default_model_config(model_family),
            }
        )
        config = _merge_missing_defaults(loaded_config, default_config)
    else:
        default_config = _to_config_node(
            {
                **base_default_config,
                "model_config": _build_default_model_config("flow_matching"),
            }
        )
        config = default_config
        save_json_config(config_path, config)

    return config
