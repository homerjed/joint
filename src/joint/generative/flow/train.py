import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float, PRNGKeyArray


def get_time(
    key: PRNGKeyArray,
    n: int,
    t0: float,
    t1: float,
    *,
    distribution: str = "uniform",
    beta_alpha: float = 1.0,
    beta_beta: float = 1.0,
    stratified: bool = True,
) -> Float[Array, "n"]:
    """Sample time points between ``t0`` and ``t1``."""
    if distribution == "uniform":
        if stratified:
            u = jr.uniform(key, (n,), minval=0.0, maxval=1.0 / n)
            u = u + jnp.arange(n) / n
        else:
            u = jr.uniform(key, (n,), minval=0.0, maxval=1.0)
    elif distribution == "beta":
        u = jr.beta(key, beta_alpha, beta_beta, shape=(n,))
    else:
        raise ValueError(f"Unknown time sampling distribution: {distribution!r}")

    return t0 + (t1 - t0) * u


def _loss_vs_t_bins(
    losses: Float[Array, "n"],
    t: Float[Array, "n"],
    *,
    t0: float,
    t1: float,
    n_bins: int,
) -> tuple[Float[Array, "b"], Float[Array, "b"]]:
    """Aggregate per-sample losses into equally spaced time bins."""
    edges = jnp.linspace(t0, t1, n_bins + 1, dtype=t.dtype)
    means = []
    counts = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i < n_bins - 1:
            mask = (t >= lo) & (t < hi)
        else:
            mask = (t >= lo) & (t <= hi)
        count = jnp.sum(mask)
        weight = mask.astype(losses.dtype)
        total = jnp.sum(losses * weight)
        mean = jnp.where(count > 0, total / count, jnp.nan)
        means.append(mean)
        counts.append(count.astype(losses.dtype))
    return jnp.stack(means), jnp.stack(counts)
