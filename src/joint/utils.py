from typing import Iterator
from collections.abc import MutableMapping
from functools import partial
from dataclasses import dataclass, field
from typing import Tuple, Literal, Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import PRNGKeyArray, Array


def count_parameters(model):
    """Count trainable array elements in an Equinox model.

    Inputs:
        model (eqx.Module): Model pytree.

    Outputs:
        int: Total number of scalar array parameters.
    """
    return sum(jax.tree.leaves(jax.tree.map(jnp.size, eqx.filter(model, eqx.is_array))))


def _get_child(obj, key):
    """Resolve a tree path key against an object or container.

    Inputs:
        obj (Any): Current tree node.
        key (jax.tree_util.*Key): Tree key object.

    Outputs:
        Any: Child object referenced by ``key``.
    """
    if isinstance(key, jax.tree_util.GetAttrKey):
        return getattr(obj, key.name)
    if isinstance(key, jax.tree_util.DictKey):
        return obj[key.key]
    if isinstance(key, jax.tree_util.SequenceKey):
        return obj[key.idx]
    if isinstance(key, jax.tree_util.FlattenedIndexKey):
        return obj[key.key]
    raise TypeError(f"Unsupported key type: {type(key)!r}")


def _resolve_path(root, path):
    """Resolve a sequence of tree keys from a root object.

    Inputs:
        root (Any): Root object.
        path (Sequence[jax.tree_util.*Key]): Tree path keys.

    Outputs:
        Any: Object at the end of the path.
    """
    x = root
    for key in path:
        x = _get_child(x, key)
    return x


def make_layernorm_decay_mask(model):
    """Build an Optax mask that disables weight decay for normalization params.

    Inputs:
        model (eqx.Module): Model containing LayerNorm/AdaLayerNorm modules.

    Outputs:
        PyTree[bool]: Boolean mask aligned with inexact parameter leaves.
    """
    params = eqx.filter(model, eqx.is_inexact_array)

    def should_decay(path, leaf):
        if leaf is None:
            return False

        parent = _resolve_path(model, path[:-1]) if path else None
        last = path[-1] if path else None

        node = model
        inside_adaln = False
        for key in path[:-1]:
            node = _get_child(node, key)
            if node.__class__.__name__ == "AdaLayerNorm":
                inside_adaln = True
                break

        is_ln_weight_or_bias = (
            isinstance(parent, eqx.nn.LayerNorm)
            and isinstance(last, jax.tree_util.GetAttrKey)
            and last.name in {"weight", "bias"}
        )

        return not (is_ln_weight_or_bias or inside_adaln)

    return jax.tree.map_with_path(should_decay, params)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["data"],
    meta_fields=[],
)
@dataclass
class ModalityInfos(MutableMapping[str, "ModalitySpec"]):
    """Dataclass-backed modality map with attribute access and validation.

    Inputs:
        data (dict[str, ModalitySpec]): Mapping from modality names to specs.

    Outputs:
        ModalityInfos: Mutable mapping enforcing key/value type and consistency checks.
    """

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
    modalities: dict[str, Array],
    n_batch: int,
    *,
    key: PRNGKeyArray,
    shuffle: bool = True,
) -> Iterator[dict[str, Array]]:
    """Yield modality-aligned mini-batches from in-memory arrays.

    Inputs:
        modalities (dict[str, Array]): Modalities sharing the same leading sample axis.
        n_batch (int): Batch size.
        key (PRNGKeyArray): Random key used for shuffling.
        shuffle (bool): Whether to reshuffle indices each epoch.

    Outputs:
        Iterator[dict[str, Array]]: Infinite iterator yielding modality batch dictionaries.
    """
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
    """Sample Gaussian noise tree with the same structure/shapes as input tree.

    Inputs:
        tree (PyTree[Array]): Reference tree.
        key (PRNGKeyArray): Random key.
        dtype (dtype | None): Optional dtype override for generated leaves.

    Outputs:
        PyTree[Array]: Noise tree with same treedef/shapes as ``tree``.
    """
    leaves, treedef = jax.tree.flatten(tree)
    keys = jr.split(key, len(leaves))
    new_leaves = [
        jr.normal(k, shape=leaf.shape, dtype=dtype or leaf.dtype)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree.unflatten(treedef, new_leaves)


def apply_ema(ema_model, model, ema_rate=0.999):
    """Apply exponential moving average update to model parameters.

    Inputs:
        ema_model (eqx.Module): Previous EMA model.
        model (eqx.Module): Current model.
        ema_rate (float): EMA keep factor in ``[0, 1)``.

    Outputs:
        eqx.Module: Updated EMA model.
    """
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
    """Specification describing one modality's data/tokenization dimensions.

    Inputs:
        name (str): Modality name.
        modality_type (Literal): One of ``image``, ``vector``, or ``graph``.
        input_shape (tuple[int, ...]): Raw modality shape.
        patch_size (int | None): Patch size for image modalities.
        sequence_length (int | None): Token count for the modality.
        embedding_dim (int | None): Token embedding width.

    Outputs:
        ModalitySpec: Validated immutable-like metadata record.
    """
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
        """Validate tokenization consistency for image-like modalities.

        Inputs:
            self.input_shape (tuple[int, ...]): ``(C, L)`` or ``(C, H, W)`` shape.
            self.patch_size (int): Patch size.

        Outputs:
            None: Updates/validates ``sequence_length`` in place.
        """
        if len(self.input_shape) not in (2, 3):
            raise ValueError(
                "Image modality requires input_shape = (C, L) or (C, H, W)."
            )

        if self.patch_size is None:
            raise ValueError("Image modality requires patch_size.")

        if len(self.input_shape) == 3:
            _, H, W = self.input_shape
            if H % self.patch_size != 0 or W % self.patch_size != 0:
                raise ValueError(
                    "patch_size must evenly divide image height and width."
                )
            expected_num_patches = (H // self.patch_size) * (W // self.patch_size)
        else:
            _, L = self.input_shape
            if L % self.patch_size != 0:
                raise ValueError("patch_size must evenly divide 1D image length.")
            expected_num_patches = L // self.patch_size

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
        """Validate vector modality configuration.

        Inputs:
            self.input_shape (tuple[int, ...]): Vector shape ``(D,)``.

        Outputs:
            None: Ensures vector constraints and default sequence length.
        """
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
        """Validate graph modality configuration.

        Inputs:
            self.input_shape (tuple[int, ...]): Graph shape ``(num_nodes, node_dim)``.

        Outputs:
            None: Ensures graph constraints and default sequence length.
        """
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
