import jax
import jax.numpy as jnp


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
    """Sample a 2D Gaussian random field with a power-law spectrum."""
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(n, d=L / n)
    ky = 2.0 * jnp.pi * jnp.fft.rfftfreq(n, d=L / n)

    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")
    k = jnp.sqrt(kx_grid**2 + ky_grid**2)

    pk = A * jnp.power((jnp.maximum(k, eps) / k0), B)

    key_re, key_im = jax.random.split(key, 2)
    re = jax.random.normal(key_re, shape=pk.shape)
    im = jax.random.normal(key_im, shape=pk.shape)
    fourier_modes = (re + 1j * im) * jnp.sqrt(pk / 2.0)
    fourier_modes = fourier_modes.at[0, 0].set(zero_mode + 0.0j)

    field = jnp.fft.irfftn(fourier_modes, s=(n, n))
    field = field * jnp.sqrt(n * n)
    return field + mean


def gaussian_to_lognormal_delta(g: jnp.ndarray, axis=None, clip=None) -> jnp.ndarray:
    """Map a Gaussian field to a zero-mean lognormal overdensity field."""
    g0 = g - jnp.mean(g, axis=axis, keepdims=True)
    var = jnp.var(g0, axis=axis, keepdims=True)

    if clip is not None:
        g0 = jnp.clip(g0, -clip, clip)

    return jnp.exp(g0 - 0.5 * var) - 1.0


def gaussian_to_lognormal_positive(g: jnp.ndarray, axis=None, clip=None) -> jnp.ndarray:
    """Map a Gaussian field to a strictly positive lognormal field."""
    g0 = g - jnp.mean(g, axis=axis, keepdims=True)
    var = jnp.var(g0, axis=axis, keepdims=True)

    if clip is not None:
        g0 = jnp.clip(g0, -clip, clip)

    return jnp.exp(g0 - 0.5 * var)
