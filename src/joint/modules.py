from typing import Optional, Callable
from functools import partial
from math import prod
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray
from einops import rearrange

from .utils import ModalityInfos, ModalitySpec


class PatchEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    patch_size: int
    position_embedding: Array

    def __init__(
        self,
        input_channels: int,
        output_shape: int,
        patch_size: int,
        num_patches: int,
        embedding_dim: int,
        key: PRNGKeyArray
    ):
        self.patch_size = patch_size

        key_linear, key_embedding = jr.split(key)

        self.linear = eqx.nn.Linear(
            self.patch_size ** 2 * input_channels,
            output_shape,
            key=key_linear
        )

        # NOTE: pos embed had extra dim for cls_token
        self.position_embedding = jr.normal(key_embedding, (num_patches, embedding_dim)) * 0.02

    def __call__(
        self, 
        x: Float[Array, "c h w"],
        position_embed: bool = True
    ) -> Float[Array, "s e"]:
        x = rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )

        x = jax.vmap(self.linear)(x)

        if position_embed:
            x = x + self.position_embedding

        return x


class ParameterEmbedding(eqx.Module):
    linear: eqx.nn.Linear
    modality_embedding: Float[Array, "e"]

    def __init__(
        self,
        input_shape: int,
        embedding_dim: int,
        key: PRNGKeyArray
    ):
        self.linear = eqx.nn.Linear(input_shape, embedding_dim, key=key)
        self.modality_embedding = jr.normal(key, (embedding_dim,)) * 0.02

    def __call__(self, p: Float[Array, "p"]) -> Float[Array, "e"]:
        return self.linear(p) + self.modality_embedding


class ParameterDecoder(eqx.Module):
    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear

    def __init__(self, embedding_dim: int, dim: int, key: PRNGKeyArray):
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.linear = eqx.nn.Linear(embedding_dim, dim, key=key)

    def __call__(self, p: Float[Array, "1 e"]) -> Float[Array, "p"]:
        return self.linear(self.norm(jnp.squeeze(p, axis=0))) # Remove axis from sequence stacking


class ImageDecoder(eqx.Module):
    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear
    unpatchify: Callable

    def __init__(self, embedding_dim, dim, patch_size, image_shape, key):
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.linear = eqx.nn.Linear(embedding_dim, dim, key=key)
        self.unpatchify = get_unpatchify(image_shape, patch_size) 

    def __call__(self, x: Float[Array, "s e"]) -> Float[Array, "c h w"]:
        x = jax.vmap(self.norm)(x)
        x = jax.vmap(self.linear)(x)
        x = self.unpatchify(x)
        return x


def _get_modality_encoder_module(name, modality_infos, key):
    modality = modality_infos[name]
    if modality.modality_type == "image":
        n_channels = modality.input_shape[0]
        module = PatchEmbedding(
            input_channels=n_channels, 
            output_shape=modality.embedding_dim, 
            patch_size=modality.patch_size,
            embedding_dim=modality.embedding_dim,
            num_patches=modality.sequence_length,
            key=key
        )
    if modality.modality_type == "vector":
        dim = prod(modality.input_shape)
        module = ParameterEmbedding(dim, modality.embedding_dim, key=key)
    return module
    

def unpatchify(x, *, image_shape, patch_size):
    c, H, W = image_shape
    ph = pw = patch_size
    h = H // ph
    w = W // pw
    assert x.shape[0] == h * w, (x.shape, h, w)
    assert x.shape[1] == c * ph * pw, (x.shape, c, ph, pw)

    return rearrange(
        x,
        "(h w) (c ph pw) -> c (h ph) (w pw)",
        h=h, w=w, c=c, ph=ph, pw=pw
    )


def get_unpatchify(image_shape, patch_size):
    return partial(unpatchify, image_shape=image_shape, patch_size=patch_size)


def _get_modality_decoder_module(name, modality_infos, key):
    # Assign decoding modules to corresponding modalities 
    # - possibly assign by 'modality_infos.modality_type == "image"' or something
    modality = modality_infos[name]
    if modality.modality_type == "image":
        n_channels = modality_infos[name].input_shape[0]
        dim = modality.patch_size ** 2 * n_channels 
        module = ImageDecoder(
            modality.embedding_dim, 
            dim, 
            image_shape=modality.input_shape, 
            patch_size=modality.patch_size, 
            key=key
        )
    if modality.modality_type == "vector":
        dim = prod(modality.input_shape)
        module = ParameterDecoder(modality.embedding_dim, dim, key=key)
    return module


def _get_encoders(encoders, keys):
    return dict(zip(keys, [encoders[key] for key in keys]))


def _get_decoders(decoders, keys):
    return dict(zip(keys, [decoders[key] for key in keys]))


class ModalitiesEncoder(eqx.Module):
    encoders: dict[str, eqx.Module]

    def __init__(
        self, 
        modality_infos: ModalityInfos, 
        *, 
        key: PRNGKeyArray
    ):
        self.encoders = dict(
            zip(
                sorted(modality_infos.keys()),
                [
                    _get_modality_encoder_module(
                        _key, 
                        modality_infos,
                        key=jr.fold_in(key, k)
                    )
                    for k, _key in enumerate(sorted(modality_infos.keys()))
                ]
            )
        )

    def __call__(self, d: dict[str, Float[Array, "..."]], stacked=False) -> dict[str, Float[Array, "..."]]:
        e = jax.tree.map(
            lambda d, e: jnp.atleast_2d(e(d)), 
            d, 
            _get_encoders(self.encoders, keys=d.keys())
        )
        if stacked:
            e = jnp.concatenate(jax.tree.leaves(e))
        return e


class ModalitiesDecoder(eqx.Module):
    decoders: dict[str, eqx.Module | ParameterDecoder]

    def __init__(
        self, 
        modality_infos: ModalityInfos, 
        *, 
        key: PRNGKeyArray
    ):
        self.decoders = dict(
            zip(
                sorted(modality_infos.keys()),
                [
                    _get_modality_decoder_module(
                        _key, 
                        modality_infos,
                        key=jr.fold_in(key, k)
                    )
                    for k, _key in enumerate(sorted(modality_infos.keys()))
                ]
            )
        )

    def __call__(self, e: dict[str, Float[Array, "..."]]) -> dict[str, Float[Array, "..."]]:
        d = jax.tree.map(
            lambda e, d: d(e),
            e, 
            _get_decoders(self.decoders, keys=e.keys())
        )
        return d