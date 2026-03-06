from typing import Iterator
from collections.abc import MutableMapping
from functools import partial
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["data"],
    meta_fields=[],
)
@dataclass
class ModalityInfos(MutableMapping[str, "ModalitySpec"]):
    """Dataclass-backed mapping with attribute access and runtime type checks."""

    data: dict[str, "ModalitySpec"] = field(default_factory=dict)

    def __post_init__(self):
        if not hasattr(self.data, "items"):
            raise TypeError(
                f"data must be a mapping of modality names to ModalitySpec, got {type(self.data)!r}."
            )
        self.data = dict(self.data)
        self._validate(self.data)

    def __getattr__(self, name):
        try:
            data = object.__getattribute__(self, "data")
        except KeyError as err:
            raise AttributeError(name) from err
        except AttributeError as err:
            raise AttributeError(name) from err
        try:
            return data[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self._validate_entry(key, value)
        candidate = dict(self.data)
        candidate[key] = value
        self._validate(candidate)
        self.data = candidate

    def __delitem__(self, key):
        candidate = dict(self.data)
        del candidate[key]
        self._validate(candidate)
        self.data = candidate

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def get(self, key, default=None):
        return self.data.get(key, default)

    @staticmethod
    def _validate_entry(key, value):
        if not isinstance(key, str):
            raise TypeError(f"Modality key must be a string, got {type(key)!r}.")
        if not isinstance(value, ModalitySpec):
            raise TypeError(
                "Modality value must be a ModalitySpec, "
                f"got {type(value)!r} for key '{key}'."
            )

    @staticmethod
    def _validate(data):
        seen_names = set()
        seen_spec_ids = set()
        for name, spec in data.items():
            ModalityInfos._validate_entry(name, spec)
            if spec.name != name:
                raise ValueError(
                    f"ModalitySpec.name mismatch for key '{name}': got '{spec.name}'."
                )
            if spec.name in seen_names:
                raise ValueError(f"Duplicate modality name '{spec.name}' found in modality infos.")
            seen_names.add(spec.name)

            spec_id = id(spec)
            if spec_id in seen_spec_ids:
                raise ValueError(
                    f"Duplicate ModalitySpec object reused for key '{name}'. "
                    "Each key must have its own ModalitySpec instance."
                )
            seen_spec_ids.add(spec_id)


def modality_dataloader(
    modalities: dict[str, jax.Array],
    n_batch: int,
    *,
    key: jax.Array,
    shuffle: bool = True,
) -> Iterator[dict[str, jax.Array]]:
    names = tuple(modalities.keys())
    arrays = tuple(modalities[name] for name in names)
    n_samples = len(arrays[0])
    idx_all = jnp.arange(n_samples)
    if any(len(array) != n_samples for array in arrays[1:]):
        raise ValueError("All modality arrays must have the same leading dimension.")
    if n_batch > n_samples:
        raise ValueError("n_batch must be <= number of samples.")

    while True:
        if shuffle:
            key, key_perm = jr.split(key)
            perm = jr.permutation(key_perm, n_samples)
        else:
            perm = idx_all
        for start in range(0, n_samples - n_batch + 1, n_batch):
            idx = perm[start : start + n_batch]
            yield {name: array[idx] for name, array in zip(names, arrays)}


def tree_random_normal_like(tree, key, dtype=None):
    leaves, treedef = jax.tree.flatten(tree)
    keys = jr.split(key, len(leaves))
    new_leaves = [
        jr.normal(k, shape=leaf.shape, dtype=dtype or leaf.dtype)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


def apply_ema(ema_model, model, ema_rate=0.999):
    m_, _m = eqx.partition(model, eqx.is_inexact_array)
    e_, _e = eqx.partition(ema_model, eqx.is_inexact_array)
    ema_fn = lambda p_ema, p: p_ema * ema_rate + p * (1. - ema_rate)
    e_ = jax.tree.map(ema_fn, e_, m_)
    return eqx.combine(e_, _m)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[
        "name",
        "modality_type",
        "input_shape",
        "patch_size",
        "sequence_length",
        "embedding_dim"
    ]
)
@dataclass
class ModalitySpec:
    name: str
    modality_type: Literal["image", "vector", "graph"]

    # Raw input space
    input_shape: Tuple[int, ...]

    # Tokenization
    patch_size: Optional[int] = None
    sequence_length: Optional[int] = None

    # Embedding space
    embedding_dim: Optional[int] = None

    def __post_init__(self):
        # ---------- Basic checks ----------
        if not isinstance(self.input_shape, tuple):
            raise TypeError("input_shape must be a tuple of ints.")

        if any(d <= 0 for d in self.input_shape):
            raise ValueError("All dimensions in input_shape must be positive.")

        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        # ---------- Modality-specific constraints ----------
        if self.modality_type == "image":
            self._validate_image()
        elif self.modality_type == "vector":
            self._validate_vector()
        elif self.modality_type == "graph":
            self._validate_graph()
        else:
            raise ValueError(f"Unknown modality_type: {self.modality_type}")

        # ---------- Generic consistency ----------
        if self.sequence_length is not None and self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")

    # ======================================================
    # Image modality checks
    # ======================================================

    def _validate_image(self):
        if len(self.input_shape) != 3:
            raise ValueError(
                "Image modality requires input_shape = (C, H, W)."
            )

        if self.patch_size is None:
            raise ValueError("Image modality requires patch_size.")

        C, H, W = self.input_shape

        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                "patch_size must evenly divide image height and width."
            )

        expected_num_patches = (H // self.patch_size) * (W // self.patch_size)

        if self.sequence_length is not None:
            if self.sequence_length != expected_num_patches:
                raise ValueError(
                    f"num_patches inconsistent. "
                    f"Expected {expected_num_patches}, "
                    f"got {self.sequence_length}."
                )
        else:
            self.sequence_length = expected_num_patches

        if self.sequence_length is not None:
            if self.sequence_length != expected_num_patches:
                raise ValueError(
                    "sequence_length must equal num_patches for image modality."
                )
        else:
            self.sequence_length = expected_num_patches

    # ======================================================
    # Vector modality checks
    # ======================================================

    def _validate_vector(self):
        if len(self.input_shape) != 1:
            raise ValueError(
                "Vector modality requires input_shape = (D,)."
            )

        if self.patch_size is not None:
            raise ValueError(
                "Vector modality should not define patch_size."
            )

        if self.sequence_length is None:
            # Treat full vector as single token
            self.sequence_length = 1

    # ======================================================
    # Graph modality checks
    # ======================================================

    def _validate_graph(self):
        if len(self.input_shape) != 2:
            raise ValueError(
                "Graph modality expects input_shape = (num_nodes, node_feat_dim)."
            )

        num_nodes, _ = self.input_shape

        if self.sequence_length is None:
            self.sequence_length = num_nodes

        if self.patch_size is not None or self.sequence_length is not None:
            raise ValueError(
                "Graph modality should not define patch_size or num_patches."
            )


def grf2d_powerlaw(
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
    # Frequencies (cycles per unit length) -> convert to angular wavenumbers (2π * freq)
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)      # shape (n,)
    ky = 2.0 * jnp.pi * jnp.fft.rfftfreq(n, d=L / n)     # shape (n//2 + 1,)

    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")         # shapes (n, n//2+1)
    k = jnp.sqrt(KX**2 + KY**2)

    # Power spectrum
    # Avoid k=0 issues with eps; we'll override the exact zero mode anyway.
    Pk = A * jnp.power((jnp.maximum(k, eps) / k0), B)

    # Draw complex Gaussian coefficients with variance ~Pk:
    # F = (N(0,1)+iN(0,1)) * sqrt(Pk/2)
    key_re, key_im = jax.random.split(key, 2)
    re = jax.random.normal(key_re, shape=Pk.shape)
    im = jax.random.normal(key_im, shape=Pk.shape)
    F = (re + 1j * im) * jnp.sqrt(Pk / 2.0)

    # Set k=0 mode explicitly (mean control)
    F = F.at[0, 0].set(zero_mode + 0.0j)

    # Inverse FFT to real space (JAX uses numpy-style normalization: irfftn includes 1/N)
    field = jnp.fft.irfftn(F, s=(n, n))

    # Optional: normalize overall scaling to be less "FFT-convention dependent".
    # This makes typical field variance scale more directly with A when changing n.
    # If you prefer raw FFT convention, remove the * (n*n)**0.5 factor.
    field = field * jnp.sqrt(n * n)

    return field + mean


def gaussian_to_lognormal_delta(g: jnp.ndarray, axis=None, clip=None) -> jnp.ndarray:
    # (Optional) enforce zero-mean before mapping; helps if your GRF has residual mean
    g0 = g - jnp.mean(g, axis=axis, keepdims=True)
    var = jnp.var(g0, axis=axis, keepdims=True)

    if clip is not None:
        g0 = jnp.clip(g0, -clip, clip)

    return jnp.exp(g0 - 0.5 * var) - 1.0


def gaussian_to_lognormal_positive(g: jnp.ndarray, axis=None, clip=None) -> jnp.ndarray:
    g0 = g - jnp.mean(g, axis=axis, keepdims=True)
    var = jnp.var(g0, axis=axis, keepdims=True)

    if clip is not None:
        g0 = jnp.clip(g0, -clip, clip)

    return jnp.exp(g0 - 0.5 * var)
