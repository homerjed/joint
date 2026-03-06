from typing import Optional
from pathlib import Path
import importlib
import json
import  diffrax as dfx
import optax

from .flow import OptimalTransportPath


class ConfigNode(dict):
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


def _normalize_config_schema(config):
    alias_map = (
        ("data", "data_config"),
        ("model", "model_config"),
        ("training_config", "train_config"),
        ("training", "train_config"),
        ("train", "train_config"),
        ("sampling", "sampling_config"),
    )
    for legacy_key, new_key in alias_map:
        if new_key not in config and legacy_key in config:
            config[new_key] = config.pop(legacy_key)
    return config


def _serialize_config(value):
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
    obj = importlib.import_module(module)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def _deserialize_config(value):
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
    encoded = _serialize_config(config)
    path.write_text(json.dumps(encoded, separators=(",", ":")), encoding="utf-8")


def load_json_config(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    decoded = _deserialize_config(raw)
    return _to_config_node(_normalize_config_schema(decoded))


def get_config(config_path: Optional[Path | str] = None):
    if config_path is None:
        config_path = Path("config.json")

    default_config = _to_config_node({
        "data_config": {
            "n_samples": 10_000,
            "n_channels": 1,
            "n_pix": 64,
            "parameter_shape": (2,),
        },
        "model_config": {
            "t_emb_dim": 8,
            "sigma_min": 1e-3,
            "patch_size": 4,
            "hidden_dim": 128,
            "n_heads": 8,
            "embedding_dim": 32,
            "n_blocks": 3,
            "dropout_rate": 0.0,
        },
        "train_config": {
            "n_steps": 100_000,
            "n_batch": 256,
            "opt_factory": optax.adam,
            "opt_kwargs": {"learning_rate": 5e-4},
            "use_ema": True,
            "sample_every": 1000,
        },
        "sampling_config": {
            "n_samples": 4,
            "t0": 0.0,
            "t1": 1.0,
            "dt0": 1e-2,
            "solver_factory": dfx.Heun,
            "path_factory": OptimalTransportPath,
        },
    })

    if config_path.exists():
        config = load_json_config(config_path)
    else:
        config = default_config
        save_json_config(config_path, config)

    return config
