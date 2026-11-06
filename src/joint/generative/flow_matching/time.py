from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, DTypeLike, Float, PRNGKeyArray

from ...naming import TimeEmbedder, TimeScalar


class TimeEmbeddingMLP(eqx.Module):
    """Learned timestep embedder built on sinusoidal features."""

    frequency_embedding_dim: int
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(
        self,
        frequency_embedding_dim: int,
        output_dim: int,
        *,
        key: PRNGKeyArray,
        hidden_dim: Optional[int] = None,
    ):
        hidden_dim = hidden_dim or (4 * output_dim)
        key1, key2 = jr.split(key, 2)

        self.frequency_embedding_dim = frequency_embedding_dim
        self.linear1 = eqx.nn.Linear(frequency_embedding_dim, hidden_dim, key=key1)
        self.linear2 = eqx.nn.Linear(hidden_dim, output_dim, key=key2)

    def __call__(self, t: TimeScalar) -> Float[Array, "e"]:
        emb = get_timestep_embedding(t, self.frequency_embedding_dim)
        emb = self.linear1(emb)
        emb = jax.nn.silu(emb)
        emb = self.linear2(emb)
        return emb


def get_timestep_embedding(
    timesteps: TimeScalar,
    embedding_dim: int,
    dtype: DTypeLike = jnp.float32,
) -> Float[Array, "e"]:
    """Create sinusoidal time embedding for a scalar or 1D time tensor."""
    if jnp.isscalar(timesteps):
        timesteps = jnp.array(timesteps, dtype=dtype)
        timesteps = jnp.atleast_1d(timesteps)
    assert len(timesteps.shape) == 1
    timesteps *= 1000.0
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return jnp.zeros((embedding_dim,), dtype=dtype)
    if half_dim == 1:
        emb = jnp.ones((1,), dtype=dtype)
    else:
        emb = jnp.log(10_000.0) / (half_dim - 1.0)
        emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype) * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=0)
    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, (0, 1))
    assert emb.shape == (embedding_dim,)
    return emb


def get_time_embedder(
    embedding_dim: int,
    *,
    t_emb_dim: Optional[int] = None,
    key: Optional[PRNGKeyArray] = None,
) -> TimeEmbedder:
    """Build a timestep embedder for transformer conditioning."""
    t_emb_dim = embedding_dim if t_emb_dim is None else t_emb_dim
    if key is None:
        return partial(get_timestep_embedding, embedding_dim=embedding_dim)
    return TimeEmbeddingMLP(t_emb_dim, embedding_dim, key=key)
