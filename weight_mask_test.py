import equinox as eqx
import jax
import jax.tree_util as jtu
import optax


def _get_child(obj, key):
    # tree_map_with_path yields key objects that describe how to descend a tree:
    # GetAttrKey -> object attribute, DictKey -> dict entry,
    # SequenceKey -> list/tuple index, FlattenedIndexKey -> flattened pytree slot.
    if isinstance(key, jtu.GetAttrKey):
        return getattr(obj, key.name)
    if isinstance(key, jtu.DictKey):
        return obj[key.key]
    if isinstance(key, jtu.SequenceKey):
        return obj[key.idx]
    if isinstance(key, jtu.FlattenedIndexKey):
        return obj[key.key]
    raise TypeError(f"Unsupported key type: {type(key)}")


def _resolve_path(root, path):
    x = root
    for key in path:
        x = _get_child(x, key)
    return x


def _path_to_str(path):
    parts = []
    for key in path:
        if isinstance(key, jtu.GetAttrKey):
            parts.append(key.name)
        elif isinstance(key, jtu.DictKey):
            parts.append(str(key.key))
        elif isinstance(key, jtu.SequenceKey):
            parts.append(str(key.idx))
        elif isinstance(key, jtu.FlattenedIndexKey):
            parts.append(str(key.key))
        else:
            parts.append(str(key))
    return ".".join(parts)


def make_layernorm_decay_mask(model):
    params = eqx.filter(model, eqx.is_inexact_array)

    def should_decay(path, leaf):
        if leaf is None:
            return False

        parent = _resolve_path(model, path[:-1]) if path else None
        last = path[-1] if path else None

        is_ln_weight_or_bias = (
            isinstance(parent, eqx.nn.LayerNorm)
            and isinstance(last, jtu.GetAttrKey)
            and last.name in {"weight", "bias"}
        )
        return not is_ln_weight_or_bias

    return jtu.tree_map_with_path(should_decay, params)


def _iter_leaves_with_path(tree):
    leaves = []

    def _collect(path, leaf):
        leaves.append((path, leaf))
        return leaf

    jtu.tree_map_with_path(_collect, tree)
    return leaves


class TinyModel(eqx.Module):
    linear1: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    linear2: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm

    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.linear1 = eqx.nn.Linear(8, 16, key=k1)
        self.norm1 = eqx.nn.LayerNorm(16)
        self.linear2 = eqx.nn.Linear(16, 4, key=k3)
        self.norm2 = eqx.nn.LayerNorm(4)


if __name__ == "__main__":
    model = TinyModel(jax.random.key(0))

    params = eqx.filter(model, eqx.is_inexact_array)
    wd_mask = make_layernorm_decay_mask(model)

    # Example optimizer using the mask.
    _ = optax.adamw(learning_rate=1e-3, weight_decay=1e-4, mask=wd_mask)

    print("Parameters excluded from weight decay:")
    for (path, p_leaf), (_, m_leaf) in zip(
        _iter_leaves_with_path(params),
        _iter_leaves_with_path(wd_mask),
    ):
        if p_leaf is None:
            continue
        if not bool(m_leaf):
            print(f"  {_path_to_str(path)}")
