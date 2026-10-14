from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Float, PRNGKeyArray

from .model import FlowMatchingModel
from .path import OptimalTransportPath
from .solvers import SolutionKwargs


@eqx.filter_jit
def single_sample_fn_ode(
    model: FlowMatchingModel,
    x_0: dict[str, Float[jax.Array, "..."]],
    *,
    solution_kwargs: SolutionKwargs,
    times: Optional[Float[jax.Array, "T"]] = None,
    key: Optional[PRNGKeyArray] = None,
) -> dict[str, Float[jax.Array, "..."]]:
    del key

    def velocity(t, x, args):
        (model_instance,) = args
        return model_instance(x, t=t)

    model = eqx.nn.inference_mode(model, True)
    saveat = dfx.SaveAt(ts=times) if times is not None else dfx.SaveAt(t1=True)
    term = dfx.ODETerm(velocity)
    sol = dfx.diffeqsolve(
        term,
        solution_kwargs.solver,
        solution_kwargs.t0,
        solution_kwargs.t1,
        solution_kwargs.dt0,
        x_0,
        args=(model,),
        saveat=saveat,
    )
    x_1 = sol.ys
    return jax.tree.map(lambda x: jnp.squeeze(x, axis=0), x_1)


@eqx.filter_jit
def single_sample_modalities_fn_ode(
    model: FlowMatchingModel,
    x_0: dict[str, Float[jax.Array, "..."]],
    modalities_to_sample: Optional[list[str]] = None,
    *,
    solution_kwargs: SolutionKwargs,
    times: Optional[Float[jax.Array, "T"]] = None,
    key: Optional[PRNGKeyArray] = None,
    bridge_path: Optional[OptimalTransportPath] = None,
) -> dict[str, Float[jax.Array, "..."]]:
    if modalities_to_sample is not None:
        modalities_to_sample = frozenset(modalities_to_sample)

    if bridge_path is None:
        bridge_path = OptimalTransportPath()

    fixed_names = ()
    conditioning_noise = {}
    attention_mask = None
    if modalities_to_sample is not None:
        fixed_names = tuple(
            name for name in x_0.keys()
            if name not in modalities_to_sample
        )
        keys = tuple(sorted(x_0.keys()))
        conditioning_mask = jnp.asarray(
            [name in fixed_names for name in keys],
            dtype=jnp.bool_,
        )
        attention_mask = model.build_directional_attention_mask(
            conditioning_mask,
            keys=keys,
        )

        if key is not None and len(fixed_names) > 0:
            noise_keys = jr.split(key, len(fixed_names))
            conditioning_noise = {
                name: jr.normal(noise_key, x_0[name].shape, dtype=x_0[name].dtype)
                for name, noise_key in zip(fixed_names, noise_keys)
            }
        else:
            conditioning_noise = {
                name: jnp.zeros_like(x_0[name])
                for name in fixed_names
            }

    def velocity(t, x, args):
        (model_instance,) = args

        if modalities_to_sample is not None and len(fixed_names) > 0:
            x_model = {
                name: (
                    x[name]
                    if name in modalities_to_sample
                    else bridge_path.psi_t(t, x_0[name], conditioning_noise[name])
                )
                for name in x.keys()
            }
        else:
            x_model = x

        velocity_value = model_instance(x_model, t=t, attention_mask=attention_mask)

        if modalities_to_sample is None:
            return velocity_value

        return {
            name: (
                velocity_value[name]
                if name in modalities_to_sample
                else jnp.zeros_like(velocity_value[name])
            )
            for name in x.keys()
        }

    model = eqx.nn.inference_mode(model, True)
    saveat = dfx.SaveAt(ts=times) if times is not None else dfx.SaveAt(t1=True)
    term = dfx.ODETerm(velocity)
    sol = dfx.diffeqsolve(
        term,
        solution_kwargs.solver,
        solution_kwargs.t0,
        solution_kwargs.t1,
        solution_kwargs.dt0,
        x_0,
        args=(model,),
        saveat=saveat,
    )
    x_1 = sol.ys
    return jax.tree.map(lambda x: jnp.squeeze(x, axis=0), x_1)
