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
    d: dict[str, Float[Array, "n ..."] | Int[Array, "n ..."]],
    opt_state: optax.OptState, 
    opt: optax.GradientTransformation, 
    v_ema: Optional[eqx.Module] = None, 
    *, 
    solution_kwargs: SolutionKwargs
) -> tuple[eqx.Module, Scalar, optax.OptState, eqx.Module]:
    n_batch = len(next(iter(d.values())))
    key_t, key_z, key_step = jr.split(key, 3)

    keys = jr.split(key_step, n_batch)

    t = get_time(key_t, n_batch, t0=solution_kwargs.t0, t1=solution_kwargs.t1)
    z = tree_random_normal_like(d, key_z)

    grad_fn = eqx.filter_value_and_grad(loss_fn)
    loss, grads = grad_fn(v, path, d, z, t, keys)

    updates, opt_state = opt.update(grads, opt_state, v)
    v = eqx.apply_updates(v, updates)

    if v_ema is not None:
        v_ema = apply_ema(v_ema, v)

    return v, loss, opt_state, v_ema


@eqx.filter_jit
def evaluate(
    v: eqx.Module,
    path: OptimalTransportPath,
    key: PRNGKeyArray,
    d: dict[str, Float[Array, "n ..."] | Int[Array, "n ..."]],
    *,
    solution_kwargs: SolutionKwargs
) -> Scalar:
    n_batch = len(next(iter(d.values())))
    key_t, key_z, key_step = jr.split(key, 3)

    v = eqx.nn.inference_mode(v)

    keys = jr.split(key_step, n_batch)
    t = get_time(key_t, n_batch, t0=solution_kwargs.t0, t1=solution_kwargs.t1)
    z = tree_random_normal_like(d, key_z)

    loss = loss_fn(v, path, d, z, t, keys)

    return loss


def evaluate_validation_set(
    model,
    path,
    val_loader,
    n_val_batches: int,
    *,
    key,
    solution_kwargs: SolutionKwargs,
):
    losses = []
    for _ in range(n_val_batches):
        key, key_eval = jr.split(key)
        losses.append(
            evaluate(
                model,
                path,
                key_eval,
                next(val_loader),
                solution_kwargs=solution_kwargs,
            )
        )
    return jnp.mean(jnp.stack(losses))