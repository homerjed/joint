from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Optional
from collections.abc import Callable
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dfx
from jaxtyping import Float, Array, PRNGKeyArray, DTypeLike

from .utils import ModalityInfos

TimeScalar = Float[Array, ""] | float
TimeEmbedder = Callable[[TimeScalar], Float[Array, "e"]]


def clip_time(t, eps=1e-6):
    t = jnp.clip(t, eps, 1. - eps)
    return t


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[],
    meta_fields=[]
)
@dataclass
class OptimalTransportPath:
    # Flow matching path

    def alpha(self, t):
        t = clip_time(t)
        return t 

    def sigma(self, t):
        t = clip_time(t)
        return 1. - t 

    def p_t(self, t, x_t, x):
        loc = self.alpha(t) * x
        scale = self.sigma(t) # std, not variance
        return jax.scipy.stats.norm.logpdf(x_t, loc=loc, scale=scale)

    def psi_t(self, t, x, z):
        return self.alpha(t) * x + self.sigma(t) * z

    def d_psi_t_dt(self, t, x, z):
        # u = jax.jacfwd(self.psi_t)(t, x, z)
        return x - z

    def d_alpha_dt(self, t):
        return jax.jacfwd(self.alpha)(t)

    def d_sigma_dt(self, t):
        return jax.jacfwd(self.sigma)(t)


class SolutionKwargs(NamedTuple):
    solver: dfx.AbstractSolver
    t0: float = 0.0
    t1: float = 1.0
    dt0: float = 0.05


def get_timestep_embedding(
    timesteps: TimeScalar, 
    embedding_dim: int, 
    dtype: DTypeLike = jnp.float32
) -> Float[Array, "e"]:
    # Convert scalar timesteps to an array
    if jnp.isscalar(timesteps):
        timesteps = jnp.array(timesteps, dtype=dtype)
        timesteps = jnp.atleast_1d(timesteps)
    assert len(timesteps.shape) == 1
    timesteps *= 1000.
    half_dim = embedding_dim // 2
    emb = jnp.log(10_000.) / (half_dim - 1.)
    emb = jnp.exp(jnp.arange(half_dim, dtype=dtype) * -emb)
    emb = timesteps.astype(dtype) * emb
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=0)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = jax.lax.pad(emb, dtype(0), ((0, 0, 0), (0, 1, 0)))
    assert emb.shape == (embedding_dim,)
    return emb


def get_time_embedder(embedding_dim: int) -> TimeEmbedder:
    return partial(get_timestep_embedding, embedding_dim=embedding_dim)


@eqx.filter_jit
def single_sample_fn_ode(
    v: eqx.Module, 
    x_0: dict[str, Float[Array, "..."]], 
    solution_kwargs: SolutionKwargs, 
    times: Optional[Float[Array, "T"]] = None
) -> dict[str, Float[Array, "..."]]:

    def velocity(t, x, args):
        (v,) = args
        return v(x, t=t)
    
    v = eqx.tree_inference(v, True)

    if times is not None:
        saveat = dfx.SaveAt(ts=times)  
    else: 
        saveat = dfx.SaveAt(t1=True)

    term = dfx.ODETerm(velocity) 

    sol = dfx.diffeqsolve(
        term, 
        solution_kwargs.solver, 
        solution_kwargs.t0, 
        solution_kwargs.t1, 
        solution_kwargs.dt0, 
        x_0, 
        args=(v,),
        saveat=saveat
    )

    x_1 = sol.ys

    # Magical strange axis appearing...
    x_1 = jax.tree.map(lambda x: jnp.squeeze(x, axis=0), x_1)

    return x_1


@eqx.filter_jit
def single_sample_modalities_fn_ode(
    v: eqx.Module, 
    x_0: dict[str, Float[Array, "..."]], 
    modality_infos: ModalityInfos,
    modalities_to_sample: Optional[list[str]] = None, 
    *,
    solution_kwargs: SolutionKwargs, 
    times: Optional[Float[Array, "T"]] = None
) -> dict[str, Float[Array, "..."]]:

    if modalities_to_sample is not None:
        modalities_to_sample = frozenset(modalities_to_sample)

    # Optional attention mask for restricting which modality tokens can be attended to.
    masks = None
    if modalities_to_sample is not None:
        masks = {}
        for name in x_0.keys():
            seq_len = modality_infos[name].sequence_length
            if name not in modalities_to_sample:
                masks[name] = jnp.zeros((seq_len,), dtype=jnp.bool_)
            else:
                masks[name] = jnp.ones((seq_len,), dtype=jnp.bool_)

    def velocity(t, x, args):
        (v, masks) = args # Same mask across time
        u = v(x, t=t, masks=masks)
        if modalities_to_sample is None:
            return u
        # Set velocities of inputs that are NOT being sampled to zero
        u = {
            name: (u[name] if name in modalities_to_sample else jnp.zeros_like(u[name]))
            for name in x.keys()
        }
        # u = jax.tree.map(
        #     lambda u: u ,
        # )
        return u
    
    v = eqx.tree_inference(v, True)

    saveat = dfx.SaveAt(ts=times) if times is not None else dfx.SaveAt(t1=True)

    term = dfx.ODETerm(velocity) 
    sol = dfx.diffeqsolve(
        term, 
        solution_kwargs.solver, 
        solution_kwargs.t0, 
        solution_kwargs.t1, 
        solution_kwargs.dt0, 
        x_0, 
        args=(v, masks),
        saveat=saveat
    )
    
    x_1 = sol.ys

    # Magical strange axis appearing...
    x_1 = jax.tree.map(
        lambda x: jnp.squeeze(x, axis=0),
        x_1
    )

    return x_1
