from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp


def clip_time(t, eps=1e-6):
    """Clamp continuous time away from singular endpoints."""
    return jnp.clip(t, eps, 1.0 - eps)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[],
)
@dataclass
class OptimalTransportPath:
    """Linear optimal-transport interpolation path used in flow matching."""

    def alpha(self, t):
        t = clip_time(t)
        return t

    def sigma(self, t):
        t = clip_time(t)
        return 1.0 - t

    def p_t(self, t, x_t, x):
        loc = self.alpha(t) * x
        scale = self.sigma(t)
        return jax.scipy.stats.norm.logpdf(x_t, loc=loc, scale=scale)

    def psi_t(self, t, x, z):
        return self.alpha(t) * x + self.sigma(t) * z

    def d_psi_t_dt(self, t, x, z):
        del t
        return x - z

    def d_alpha_dt(self, t):
        return jax.jacfwd(self.alpha)(t)

    def d_sigma_dt(self, t):
        return jax.jacfwd(self.sigma)(t)
