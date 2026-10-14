from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray, Scalar, Bool, Int


def _zero_init_linear(linear: eqx.nn.Linear) -> eqx.nn.Linear:
    """Zero-initialize an Equinox linear layer."""
    linear = eqx.tree_at(
        lambda m: m.weight,
        linear,
        jnp.zeros_like(linear.weight),
    )
    linear = eqx.tree_at(
        lambda m: m.bias,
        linear,
        jnp.zeros_like(linear.bias),
    )
    return linear


class AdaLayerNorm(eqx.Module):
    """LayerNorm with optional external scale/shift modulation.

    Inputs:
        embed_dim (int): Feature dimension for normalized tokens.
        eps (float): Numerical epsilon for layer normalization.

    Outputs:
        Array: Normalized token with optional condition-dependent scale/shift.
    """
    ln: eqx.nn.LayerNorm

    def __init__(
        self, 
        embed_dim: int, 
        *, 
        eps: float = 1e-5
    ):
        self.ln = eqx.nn.LayerNorm(
            shape=embed_dim, eps=eps, use_bias=False, use_weight=False
        )

    def __call__(
        self, 
        x: Float[Array, "x"], 
        *,
        scale: Optional[Float[Array, "x"]] = None,
        shift: Optional[Float[Array, "x"]] = None,
    ) -> Float[Array, "x"]:
        """Apply adaptive normalization.

        Inputs:
            x (Array): Input token embedding.
            scale (Array | None): Optional feature-wise multiplicative modulation.
            shift (Array | None): Optional feature-wise additive modulation.

        Outputs:
            Array: Adaptively normalized token.
        """
        x = self.ln(x)
        if scale is None or shift is None:
            return x
        return x * (1. + scale) + shift


class AttentionBlock(eqx.Module):
    """Transformer residual block with adaLN-zero modulation.

    Inputs:
        input_shape (int): Token embedding size.
        hidden_dim (int): MLP hidden size.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout probability.
        key (PRNGKeyArray): Random key for parameter initialization.

    Outputs:
        Array: Updated token sequence with same shape as input.
    """
    layer_norm1: AdaLayerNorm
    layer_norm2: AdaLayerNorm
    attention: eqx.nn.MultiheadAttention
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    modulation: eqx.nn.Linear
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
        keys = jr.split(key, 6)

        self.layer_norm1 = AdaLayerNorm(input_shape)
        self.layer_norm2 = AdaLayerNorm(input_shape)
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=keys[2])

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=keys[3])
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=keys[4])
        self.modulation = _zero_init_linear(
            eqx.nn.Linear(input_shape, 6 * input_shape, key=keys[5])
        )
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: Float[Array, "s e"],
        c: Optional[Float[Array, "e"]] = None,
        mask: Optional[Bool[Array, "s s"]] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "s e"]:
        """Run one attention + MLP residual block.

        Inputs:
            x (Array): Token sequence with shape ``(S, E)``.
            c (Array | None): Conditioning embedding.
            mask (Array | None): Attention mask with shape ``(S, S)``.
            key (PRNGKeyArray | None): Dropout key.

        Outputs:
            Array: Updated token sequence.
        """
        if c is None:
            shift_msa = scale_msa = shift_mlp = scale_mlp = None
            gate_msa = gate_mlp = 1.0
        else:
            modulation = self.modulation(jax.nn.silu(c))
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
                modulation,
                6,
            )

        input_x = jax.vmap(
            lambda token: self.layer_norm1(token, scale=scale_msa, shift=shift_msa)
        )(x)
        input_x = self.attention(input_x, input_x, input_x, mask=mask)
        x = x + gate_msa * input_x

        input_x = jax.vmap(
            lambda token: self.layer_norm2(token, scale=scale_mlp, shift=shift_mlp)
        )(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        if key is not None:
            key1, key2 = jr.split(key)
        else:
            key1 = key2 = None

        input_x = self.dropout1(input_x, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, key=key2)

        x = x + gate_mlp * input_x

        return x


class VisionTransformer(eqx.Module):
    """Stack of attention blocks for sequence processing.

    Inputs:
        embedding_dim (int): Token embedding size.
        hidden_dim (int): MLP hidden size.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of residual blocks.
        dropout_rate (float): Dropout probability.
        key (PRNGKeyArray): Random key for parameter initialization.

    Outputs:
        Array: Updated token sequence or class token depending on call mode.
    """
    cls_token: Float[Array, "1 e"]
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

    def __call__(
        self,
        x: Float[Array, "s e"],
        # NOTE: Embed condition variables (e.g. C in G|C) here and sum them together
        # this means that the correct PDF is being sampled, and C doesn't always have
        # to be present in general. `c` here would be the embedded modality, so as to 
        # resuse existing modules.
        c: Optional[Float[Array, "e"]], 
        mask: Optional[Bool[Array, "s s"]] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        use_class_token: bool = False # May want to return class token for downstream regression
    ) -> Float[Array, "s e"] | Float[Array, "1 e"]:
        """Run transformer forward pass on token sequence.

        Inputs:
            x (Array): Input token sequence.
            c (Array | None): Conditioning embedding.
            mask (Array | None): Attention mask.
            key (PRNGKeyArray | None): Dropout/attention randomness key.
            use_class_token (bool): If ``True``, prepend and return class token.

        Outputs:
            Array: Token sequence output or class token output.
        """

        if use_class_token:
            x = jnp.concatenate([self.cls_token, x])
            if mask is not None:
                mask = jnp.pad(mask, ((1, 0), (1, 0)), constant_values=True)

        if key is not None:
            dropout_key, *attention_keys = jr.split(key, len(self.attention_blocks) + 1)
        else:
            dropout_key = None
            attention_keys = [None] * len(self.attention_blocks)

        x = self.dropout(x, key=dropout_key)

        for block, attention_key in zip(self.attention_blocks, attention_keys):
            x = block(x, c=c, mask=mask, key=attention_key)

        if use_class_token:
            x = x[0] # Select the CLS token.

        return x
