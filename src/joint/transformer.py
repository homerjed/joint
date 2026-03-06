from typing import Optional, Sequence, Callable
from itertools import accumulate
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, Scalar, Bool

from .modules import ModalitiesEncoder, ModalitiesDecoder
from .utils import ModalitySpec

Modalities = dict[str, Array]


class AdaLayerNorm(eqx.Module):
    ln: eqx.nn.LayerNorm
    scale_proj: eqx.nn.Linear
    shift_proj: eqx.nn.Linear

    def __init__(
        self, 
        embed_dim: int, 
        *, 
        key: PRNGKeyArray, 
        eps=1e-5
    ):
        keys = jr.split(key)

        self.ln = eqx.nn.LayerNorm(
            shape=embed_dim, eps=eps, use_bias=False, use_weight=False
        )

        # Output shape (dim,) and zero init
        self.scale_proj = eqx.nn.Linear(
            embed_dim, embed_dim, key=keys[0], use_bias=True
        )
        self.shift_proj = eqx.nn.Linear(
            embed_dim, embed_dim, key=keys[1], use_bias=True
        )

        # Zero init for scale and shift
        self.scale_proj = eqx.tree_at(
            lambda m: m.weight, 
            self.scale_proj, 
            jnp.zeros_like(self.scale_proj.weight)
        )
        self.scale_proj = eqx.tree_at(
            lambda m: m.bias, 
            self.scale_proj, 
            jnp.zeros_like(self.scale_proj.bias)
        )

        self.shift_proj = eqx.tree_at(
            lambda m: m.weight, 
            self.shift_proj, 
            jnp.zeros_like(self.shift_proj.weight)
        )
        self.shift_proj = eqx.tree_at(
            lambda m: m.bias, 
            self.shift_proj, 
            jnp.zeros_like(self.shift_proj.bias)
        )

    def __call__(
        self, 
        x: Float[Array, "x"], 
        c: Float[Array, "c"]
    ) -> Float[Array, "x"]:
        if c is None:
            return self.ln(x)
        scale = self.scale_proj(c)
        shift = self.shift_proj(c)
        return self.ln(x) * (1. + scale) + shift


class AttentionBlock(eqx.Module):
    layer_norm1: AdaLayerNorm
    layer_norm2: AdaLayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, 5)

        self.layer_norm1 = AdaLayerNorm(input_shape, key=keys[0])
        self.layer_norm2 = AdaLayerNorm(input_shape, key=keys[1])
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=keys[2])

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=keys[3])
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=keys[4])
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "s e"],
        c: Optional[Float[Array, "e"]] = None,
        mask: Optional[Float[Array, "s e"]] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "s e"]:

        input_x = jax.vmap(eqx.Partial(self.layer_norm1, c=c))(x)
        x = x + self.attention(input_x, input_x, input_x, mask=mask)

        input_x = jax.vmap(eqx.Partial(self.layer_norm2, c=c))(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        if key is not None:
            key1, key2 = jr.split(key, num=2)
        else:
            key1 = key2 = None

        input_x = self.dropout1(input_x, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, key=key2)

        x = x + input_x

        return x


class VisionTransformer(eqx.Module):
    cls_token: jnp.ndarray
    attention_blocks: list[AttentionBlock]
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, 5)

        self.cls_token = jr.normal(keys[0], (1, embedding_dim))

        self.attention_blocks = [
            AttentionBlock(
                embedding_dim, hidden_dim, num_heads, dropout_rate, key=_key
            )
            for _key in jr.split(keys[1], num_layers)
        ]

        self.dropout = eqx.nn.Dropout(dropout_rate)

        # No MLP now but maybe output layernorm?

    def __call__(
        self,
        x: Float[Array, "s e"],
        c: Optional[Float[Array, "e"]],
        mask: Optional[Float[Array, "s e"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        use_class_token: bool = False # May want to return class token for downstream regression
    ) -> Float[Array, "s e"] | Float[Array, "1 e"]:

        if use_class_token:
            x = jnp.concatenate([self.cls_token, x])

        if key is not None:
            dropout_key, *attention_keys = jr.split(key, num=len(self.attention_blocks) + 1)
        else:
            dropout_key = None
            attention_keys = [None] * len(self.attention_blocks)

        x = self.dropout(x, key=dropout_key)

        for block, attention_key in zip(self.attention_blocks, attention_keys):
            x = block(x, c=c, mask=mask, key=attention_key)

        if use_class_token:
            x = x[0] # Select the CLS token.

        return x


class FlowTransformer(eqx.Module):
    modalities_encoder: ModalitiesEncoder
    modalities_decoder: ModalitiesDecoder
    transformer: VisionTransformer
    time_embedder: eqx.Module | Callable
    modality_infos: dict[str, ModalitySpec]

    def __init__(
        self, 
        modalities_encoder: ModalitiesEncoder, 
        modalities_decoder: ModalitiesDecoder, 
        transformer: VisionTransformer,
        time_embedder: eqx.Module,
        modality_infos: dict[str, ModalitySpec]
    ):
        self.modalities_encoder = modalities_encoder
        self.modalities_decoder = modalities_decoder
        self.transformer = transformer
        self.time_embedder = time_embedder
        self.modality_infos = modality_infos

    def get_modality_lengths(self, keys: Optional[Sequence[str]] = None) -> list[int]:
        if keys is None:
            keys = sorted(self.modality_infos.keys())
        return [self.modality_infos[k].sequence_length for k in keys]

    def get_modalities_length(self, keys: Optional[Sequence[str]] = None) -> list[int]:
        # list(accumulate(self.get_modality_lengths(keys=sorted(d.keys()))[:-1]))
        lengths = self.get_modality_lengths(keys)[:-1] # Don't count last one
        length = list(accumulate(lengths))
        return length

    def __call__(
        self, 
        d: Modalities, 
        t: Optional[Scalar] = None, 
        # condition: Optional[Float[Array, "..."]] = None, # Optional input: must be embedding dim?
        mask: Optional[Bool[Array, "..."]] = None,
        key: Optional[PRNGKeyArray] = None,
        *,
        use_class_token: bool = False
    ) -> Modalities | Float[Array, "1 e"]:

        keys = sorted(d.keys())

        d = {k: d[k] for k in keys} # normalize modality order once

        # Embed time as a condition into attention / MLP / LayerNorms
        # not as a token!
        if t is not None:
            c = self.time_embedder(t)
        else:
            c = None

        # Record input shapes for use in reconstructing the output
        # modalities that are present or not
        e = self.modalities_encoder(d)

        # Stack modalities and input to transformer
        e = jnp.concatenate(jax.tree.leaves(e))
        h = self.transformer(e, c, mask=mask, key=key, use_class_token=use_class_token)
        
        lengths = self.get_modalities_length(keys) 
        h = jax.tree.unflatten(jax.tree.structure(d), jnp.split(h, lengths))
        # h = {k: x for k, x in zip(keys, h)} # same order

        v = self.modalities_decoder(h)

        return v


        # v = jax.vmap(self.transformer.mlp)(h) # NOTE: not necessary?
        # Get sizes to split v into each modality
        # dims = jax.tree.map(lambda s: s[0], shapes, is_leaf=lambda l: isinstance(l, tuple))
        # lengths = jnp.cumsum(jnp.array(jax.tree.leaves(dims)))[:-1]
        # v = jnp.split(v, lengths) # Should accumulate dims?
        # v = 
        # v = jax.tree.unflatten(jax.tree.structure(d), v)