from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray

from .model import SphereEncoderModel


@eqx.filter_jit
def single_sample_modalities_fn(
    model: SphereEncoderModel,
    x_0: dict[str, jax.Array],
    modalities_to_sample: Optional[list[str]] = None,
    *,
    forward_steps: int = 1,
    use_sampling_scheduler: bool = False,
    cache_sampling_noise: bool = False,
    key: PRNGKeyArray,
) -> dict[str, jax.Array]:
    if modalities_to_sample is not None:
        modalities_to_sample = frozenset(modalities_to_sample)

    ordered_keys = tuple(sorted(x_0.keys()))
    fixed_names = ()
    attention_mask = None
    if modalities_to_sample is not None:
        fixed_names = tuple(
            name for name in ordered_keys
            if name not in modalities_to_sample
        )
        conditioning_mask = jnp.asarray(
            [name in fixed_names for name in ordered_keys],
            dtype=jnp.bool_,
        )
        attention_mask = model.build_directional_attention_mask(
            conditioning_mask,
            keys=ordered_keys,
        )

    key_latents, key_loop = jr.split(key)
    latents = jr.normal(
        key_latents,
        (model.tokenizer.total_sequence_length(), model.latent_dim),
    )
    latents = model.spherify(latents)

    if fixed_names:
        key_loop, key_encode = jr.split(key_loop)
        conditioning_inputs = {
            name: (x_0[name] if name in fixed_names else jnp.zeros_like(x_0[name]))
            for name in ordered_keys
        }
        conditioning_latents = model.spherify(
            model.encode(conditioning_inputs, attention_mask=attention_mask, key=key_encode)
        )
        token_owners = model.tokenizer.get_modality_token_owners(keys=ordered_keys)
        fixed_tokens = conditioning_mask[token_owners, None]
        # Observations must reach the decoder before the first generated sample.
        latents = jnp.where(fixed_tokens, conditioning_latents, latents)

    shared_noise = None
    if cache_sampling_noise:
        key_loop, key_shared = jr.split(key_loop)
        shared_noise = jr.normal(key_shared, latents.shape, dtype=latents.dtype)

    key_loop, key_decode = jr.split(key_loop)
    current_modalities = model.decode(latents, attention_mask=attention_mask, key=key_decode)
    if modalities_to_sample is not None and len(fixed_names) > 0:
        current_modalities = {
            name: (x_0[name] if name in fixed_names else current_modalities[name])
            for name in ordered_keys
        }

    if forward_steps <= 1:
        return current_modalities

    for step in range(forward_steps - 1):
        key_loop, key_encode, key_noise, key_decode = jr.split(key_loop, 4)
        latents = model.encode(
            current_modalities,
            attention_mask=attention_mask,
            key=key_encode,
        )
        latents = model.spherify(latents)
        if use_sampling_scheduler:
            noise_scaler = 1.0 - float(step + 1) / float(forward_steps)
        else:
            noise_scaler = 1.0
        # radius = jnp.asarray(noise_scaler, dtype=latents.dtype).reshape((1, 1))
        radius = jnp.atleast_2d(noise_scaler).astype(latents.dtype)
        latents = model.add_noise(
            key_noise,
            latents,
            radius,
            noise=shared_noise,
        )
        current_modalities = model.decode(
            latents,
            attention_mask=attention_mask,
            key=key_decode,
        )
        if modalities_to_sample is not None and len(fixed_names) > 0:
            current_modalities = {
                name: (x_0[name] if name in fixed_names else current_modalities[name])
                for name in ordered_keys
            }

    return current_modalities


@eqx.filter_jit
def single_sample_fn(
    model: SphereEncoderModel,
    x_0: dict[str, jax.Array],
    *,
    forward_steps: int = 1,
    use_sampling_scheduler: bool = False,
    cache_sampling_noise: bool = False,
    key: PRNGKeyArray,
) -> dict[str, jax.Array]:
    return single_sample_modalities_fn(
        model,
        x_0,
        modalities_to_sample=None,
        forward_steps=forward_steps,
        use_sampling_scheduler=use_sampling_scheduler,
        cache_sampling_noise=cache_sampling_noise,
        key=key,
    )
