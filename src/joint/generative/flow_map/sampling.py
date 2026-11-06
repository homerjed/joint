import math
from numbers import Integral
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ...naming import Modalities


def _time_grid(n_steps, t0, t1, times):
    if times is None:
        if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer.")
        if not (math.isfinite(t0) and math.isfinite(t1) and 0 <= t0 < t1 <= 1):
            raise ValueError("Sampling requires 0 <= t0 < t1 <= 1.")
        return jnp.linspace(t0, t1, int(n_steps) + 1)
    times = jnp.asarray(times, dtype=jnp.float32)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a one-dimensional grid with at least two entries.")
    return eqx.error_if(
        times,
        jnp.any(~jnp.isfinite(times))
        | jnp.any(jnp.diff(times) <= 0)
        | jnp.any((times < 0) | (times > 1)),
        "times must be finite, strictly increasing, and within [0, 1].",
    )


@eqx.filter_jit
def single_sample_modalities_fn(
    model,
    x_0: Modalities,
    modalities_to_sample: Optional[Sequence[str]] = None,
    *,
    n_steps: int = 4,
    t0: float = 0.0,
    t1: float = 1.0,
    times=None,
    key: Optional[PRNGKeyArray] = None,
) -> Modalities:
    """Compose learned maps, using exactly one model call per grid interval.

    Generated modalities in x_0 contain base noise (or a state at t0).
    Other modalities contain clean observations and stay fixed at every step,
    matching conditional flow-map training. ``times`` overrides the uniform
    grid and includes both endpoints; only the final state is returned.
    Dropout is disabled, so ``key`` is accepted for the family API but unused.
    """
    del key
    names = tuple(sorted(x_0))
    if not names:
        raise ValueError("x_0 must contain at least one modality.")
    generated = frozenset(names if modalities_to_sample is None else modalities_to_sample)
    unknown = generated.difference(names)
    if unknown:
        raise ValueError(f"Unknown modalities to sample: {sorted(unknown)}")
    grid = _time_grid(n_steps, t0, t1, times)
    if not generated:
        return x_0

    model = eqx.nn.inference_mode(model, True)
    attention_mask = None
    if generated != frozenset(names):
        attention_mask = model.build_directional_attention_mask(
            jnp.asarray([name not in generated for name in names]), keys=names
        )

    def step(state, interval):
        s, t = interval
        mapped = model.flow_map(state, s, t, attention_mask=attention_mask)
        state = {
            name: mapped[name] if name in generated else x_0[name]
            for name in names
        }
        return state, None

    result, _ = jax.lax.scan(step, x_0, (grid[:-1], grid[1:]))
    return result


def single_sample_fn(
    model,
    x_0: Modalities,
    *,
    n_steps: int = 4,
    t0: float = 0.0,
    t1: float = 1.0,
    times=None,
    key: Optional[PRNGKeyArray] = None,
) -> Modalities:
    """Generate all modalities by composing one or a few learned maps."""
    return single_sample_modalities_fn(
        model, x_0, n_steps=n_steps, t0=t0, t1=t1, times=times, key=key
    )
