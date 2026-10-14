from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ...multimodal import MultimodalTokenizer
from ...naming import Modalities
from ...transformer import VisionTransformer


class SphereEncoderModel(eqx.Module):
    """Sphere-style denoising autoencoder built on multimodal DiT blocks."""

    tokenizer: MultimodalTokenizer
    encoder_transformer: VisionTransformer
    decoder_transformer: VisionTransformer
    latent_in: eqx.nn.Linear
    latent_out: eqx.nn.Linear
    latent_dim: int = eqx.field(static=True)
    noise_sigma_max_angle: float = eqx.field(static=True)

    def get_modality_lengths(self, keys: Optional[Sequence[str]] = None) -> list[int]:
        return self.tokenizer.get_modality_lengths(keys)

    def build_directional_attention_mask(
        self,
        conditioning_mask: Bool[Array, "m"],
        keys: Optional[Sequence[str]] = None,
    ) -> Bool[Array, "s s"]:
        return self.tokenizer.build_directional_attention_mask(conditioning_mask, keys)

    @staticmethod
    def spherify(latents: Float[Array, "s l"], eps: float = 1e-6) -> Float[Array, "s l"]:
        axes = tuple(range(latents.ndim))
        rms = jnp.sqrt(jnp.mean(jnp.square(latents), axis=axes, keepdims=True) + eps)
        return latents / rms

    def encode(
        self,
        modalities: Modalities,
        *,
        attention_mask: Optional[Bool[Array, "s s"]] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "s l"]:
        tokens = self.tokenizer.encode(modalities, stacked=True)
        hidden = self.encoder_transformer(tokens, None, mask=attention_mask, key=key)
        return jax.vmap(self.latent_in)(hidden)

    def decode(
        self,
        latents: Float[Array, "s l"],
        *,
        attention_mask: Optional[Bool[Array, "s s"]] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Modalities:
        hidden = jax.vmap(self.latent_out)(latents)
        hidden = self.decoder_transformer(hidden, None, mask=attention_mask, key=key)
        return self.tokenizer.decode(hidden)

    def reconstruct(
        self,
        modalities: Modalities,
        *,
        attention_mask: Optional[Bool[Array, "s s"]] = None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Modalities:
        if key is not None:
            key_encode, key_decode = jr.split(key)
        else:
            key_encode = key_decode = None
        latents = self.spherify(
            self.encode(modalities, attention_mask=attention_mask, key=key_encode)
        )
        return self.decode(latents, attention_mask=attention_mask, key=key_decode)

    def add_noise(
        self,
        key: PRNGKeyArray,
        latents: Float[Array, "s l"],
        radius: Float[Array, "..."],
        *,
        noise: Optional[Float[Array, "s l"]] = None,
    ) -> Float[Array, "s l"]:
        sigma = jnp.tan(jnp.deg2rad(self.noise_sigma_max_angle))
        if noise is None:
            noise = jr.normal(key, latents.shape, dtype=latents.dtype)
        perturbed = self.spherify(latents + radius * sigma * noise)
        return jnp.where(radius == 0, latents, perturbed)
