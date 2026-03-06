from pathlib import Path
from typing import Union, Optional
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray, Scalar, Array, Float, Int

from .utils import apply_ema, tree_random_normal_like
from .flow import SolutionKwargs, OptimalTransportPath


def save_training_state(
    checkpoint_path: Union[str, Path],
    model: eqx.Module,
    ema_model: eqx.Module,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    train_step: int,
) -> None:
    # NOTE: check handling in case use_ema=False meaning that ema_model=None
    state = {
        "model": model,
        "ema_model": ema_model,
        "optimiser": optimiser,
        "optimiser_state": optimiser_state,
        "train_step": jnp.asarray(train_step, dtype=jnp.int32),
    }
    eqx.tree_serialise_leaves(checkpoint_path, state)


def load_training_state(
    checkpoint_path: Union[str, Path],
    model,
    optimiser,
    optimiser_state,
    train_step: int = 0,
) -> tuple[
    eqx.Module, 
    eqx.Module, 
    optax.GradientTransformation, 
    optax.OptState, 
    int
]:
    template = {
        "model": model,
        "ema_model": model, # Same model structure
        "optimiser": optimiser,
        "optimiser_state": optimiser_state,
        "train_step": jnp.asarray(train_step, dtype=jnp.int32),
    }
    state = eqx.tree_deserialise_leaves(checkpoint_path, template)
    return (
        state["model"],
        state["ema_model"],
        state["optimiser"],
        state["optimiser_state"],
        int(state["train_step"]),
    )


def get_time(key, n, t0, t1):
    t = jr.uniform(key, (n,), minval=t0, maxval=t1 / n)
    t = t + (t1 / n) * jnp.arange(n)
    return t


def _mse(y_, y):
    return jnp.mean(jnp.square(jnp.subtract(y_, y)))


def loss_fn(model, path, d, z, t, keys):
    # Diffuse modalities in d according to same diffusion path

    def _single_loss_fn(t, d, z, key):
        u = jax.tree.map(lambda x, z: path.d_psi_t_dt(t, x, z), d, z)
        psi = jax.tree.map(lambda x, z: path.psi_t(t, x, z), d, z)
        u_ = model(psi, t, key=key)
        l = sum(jax.tree.leaves(jax.tree.map(_mse, u, u_)))
        return l 

    l = jnp.mean(jax.vmap(_single_loss_fn)(t, d, z, keys))

    return l


@eqx.filter_jit(donate="all")
def make_step(
    v: eqx.Module, 
    path: OptimalTransportPath, 
    key: PRNGKeyArray, 
    X: Float[Array, "n _ _ _"], # NOTE: this step function is setup for only two modalities
    Y: Float[Array, "n _"] | Int[Array, "n _"], 
    opt_state: optax.OptState, 
    opt: optax.GradientTransformation, 
    n_batch: int, 
    v_ema: Optional[eqx.Module] = None, 
    *, 
    solution_kwargs: SolutionKwargs
) -> tuple[eqx.Module, Scalar, optax.OptState, eqx.Module]:
    key_idx, key_t, key_step = jr.split(key, 3)

    keys = jr.split(key_step, n_batch)

    idx_x, idx_z = jr.choice(key_idx, len(X), (2, n_batch))
    x, y = X[idx_x], Y[idx_x]
    t = get_time(key_t, n_batch, t0=solution_kwargs.t0, t1=solution_kwargs.t1)

    d = dict(x=x, y=y) # Setup modality dictionary (NOTE: maybe one entry per model parameter?)

    z = tree_random_normal_like(d, key)

    grad_fn = eqx.filter_value_and_grad(loss_fn)
    loss, grads = grad_fn(v, path, d, z, t, keys)

    updates, opt_state = opt.update(grads, opt_state, v)
    v = eqx.apply_updates(v, updates)

    if v_ema is not None:
        v_ema = apply_ema(v_ema, v)

    return v, loss, opt_state, v_ema
