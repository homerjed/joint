from typing import Optional, Callable, Sequence
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
    """Encode 1D/2D image-like tensors into patch tokens with a small conv stem.

    Inputs:
        input_shape (tuple[int, ...]): Input shape as ``(C, L)`` or ``(C, H, W)``.
        output_shape (int): Patch embedding width.
        patch_size (int): Spatial patch size.
        num_patches (int): Number of patch tokens.
        embedding_dim (int): Positional embedding width.
        key (PRNGKeyArray): Random key for parameter initialization.

    Outputs:
        Array: Sequence of patch embeddings with shape ``(num_patches, embedding_dim)``.
    """
    stem_conv1: eqx.Module
    stem_conv2: eqx.Module
    patch_size: int
    position_embedding: Array
    spatial_ndim: int

    def __init__(
        self,
        input_shape: tuple[int, ...],
        output_shape: int,
        patch_size: int,
        num_patches: int,
        embedding_dim: int,
        key: PRNGKeyArray
    ):
        if len(input_shape) not in (2, 3):
            raise ValueError("PatchEmbedding expects input_shape as (C, L) or (C, H, W).")

        self.patch_size = patch_size
        self.spatial_ndim = len(input_shape) - 1
        spatial_shape = input_shape[1:]
        if any(length % patch_size != 0 for length in spatial_shape):
            raise ValueError(
                "PatchEmbedding expects every spatial dimension to be divisible by patch_size."
            )

        input_channels = input_shape[0]
        stem_hidden_dim = output_shape
        local_kernel_size = 7

        key_conv1, key_conv2, key_embedding = jr.split(key, 3)

        # First mix local neighborhoods at full spatial resolution, then downsample
        # to the same non-overlapping patch grid used by the decoder.
        self.stem_conv1 = eqx.nn.Conv(
            self.spatial_ndim,
            input_channels,
            stem_hidden_dim,
            kernel_size=local_kernel_size,
            padding=local_kernel_size // 2,
            key=key_conv1,
        )
        self.stem_conv2 = eqx.nn.Conv(
            self.spatial_ndim,
            stem_hidden_dim,
            output_shape,
            kernel_size=patch_size,
            stride=patch_size,
            key=key_conv2,
        )

        # NOTE: pos embed had extra dim for cls_token
        self.position_embedding = (
            jr.normal(key_embedding, (num_patches, embedding_dim)) * 0.02
        )

    def __call__(
        self, 
        x: Float[Array, "..."],
        position_embed: bool = True
    ) -> Float[Array, "s e"]:
        """Embed an image-like modality into patch tokens.

        Inputs:
            x (Array): Input tensor in ``(C, L)`` or ``(C, H, W)`` format.
            position_embed (bool): If ``True``, add learned positional embeddings.

        Outputs:
            Array: Patch token sequence with shape ``(S, E)``.
        """
        x = self.stem_conv1(x)
        x = jax.nn.gelu(x)
        x = self.stem_conv2(x)

        if self.spatial_ndim == 2:
            x = rearrange(x, "e h w -> (h w) e")
        else:
            x = rearrange(x, "e l -> l e")

        if position_embed:
            assert x.shape == self.position_embedding.shape
            x = x + self.position_embedding

        return x


class ParameterEmbedding(eqx.Module):
    """Project a vector modality into embedding space.

    Inputs:
        input_shape (int): Vector dimension.
        embedding_dim (int): Output embedding width.
        key (PRNGKeyArray): Random key for parameter initialization.

    Outputs:
        Array: Embedded vector token with shape ``(embedding_dim,)``.
    """
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
        """Embed a parameter vector.

        Inputs:
            p (Array): Vector modality input.

        Outputs:
            Array: Embedded token vector.
        """
        return self.linear(p) + self.modality_embedding


class SetEmbedding(eqx.Module):
    """Embed each set row with the same MLP, without row-position embeddings."""

    mlp: eqx.nn.MLP
    modality_embedding: Float[Array, "e"]

    def __init__(self, input_dim: int, embedding_dim: int, key: PRNGKeyArray):
        key_mlp, key_embedding = jr.split(key)
        self.mlp = eqx.nn.MLP(
            input_dim, embedding_dim, embedding_dim, depth=1,
            activation=jax.nn.gelu, key=key_mlp,
        )
        self.modality_embedding = jr.normal(key_embedding, (embedding_dim,)) * 0.02

    def __call__(self, x: Float[Array, "n f"]) -> Float[Array, "n e"]:
        return jax.vmap(self.mlp)(x) + self.modality_embedding


class SetDecoder(eqx.Module):
    """Decode each set token independently with shared parameters."""

    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear

    def __init__(self, embedding_dim: int, output_dim: int, key: PRNGKeyArray):
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.linear = eqx.nn.Linear(embedding_dim, output_dim, key=key)

    def __call__(self, x: Float[Array, "n e"]) -> Float[Array, "n f"]:
        return jax.vmap(lambda row: self.linear(self.norm(row)))(x)


class ParameterDecoder(eqx.Module):
    """Decode a single parameter token back to vector space.

    Inputs:
        embedding_dim (int): Token embedding width.
        dim (int): Output vector dimension.
        key (PRNGKeyArray): Random key for parameter initialization.

    Outputs:
        Array: Decoded vector modality.
    """
    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear

    def __init__(self, embedding_dim: int, dim: int, key: PRNGKeyArray):
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.linear = eqx.nn.Linear(embedding_dim, dim, key=key)

    def __call__(self, p: Float[Array, "1 e"]) -> Float[Array, "p"]:
        """Decode a one-token sequence into a vector.

        Inputs:
            p (Array): Sequence with one token.

        Outputs:
            Array: Decoded vector.
        """
        return self.linear(self.norm(jnp.squeeze(p, axis=0))) # Remove first axis (from sequence stacking)


class ImageDecoder(eqx.Module):
    """Decode patch tokens back into image-like tensors.

    Inputs:
        embedding_dim (int): Token embedding width.
        dim (int): Per-token decoded patch dimension.
        patch_size (int): Spatial patch size used for tokenization.
        image_shape (Sequence[int]): Target shape ``(C, L)`` or ``(C, H, W)``.
        key (PRNGKeyArray): Random key for parameter initialization.

    Outputs:
        Array: Reconstructed modality in original image-like shape.
    """
    norm: eqx.nn.LayerNorm
    linear: eqx.nn.Linear
    unpatchify: Callable

    def __init__(
        self, 
        embedding_dim: int, 
        dim: int, 
        patch_size: int, 
        image_shape: Sequence[int], 
        key: PRNGKeyArray
    ):
        self.norm = eqx.nn.LayerNorm(embedding_dim)
        self.linear = eqx.nn.Linear(embedding_dim, dim, key=key)
        self.unpatchify = get_unpatchify(image_shape, patch_size) 

    def __call__(self, x: Float[Array, "s e"]) -> Float[Array, "..."]:
        """Decode patch token sequence to original image-like layout.

        Inputs:
            x (Array): Sequence of patch tokens.

        Outputs:
            Array: Reconstructed tensor.
        """
        x = jax.vmap(self.norm)(x)
        x = jax.vmap(self.linear)(x)
        x = self.unpatchify(x)
        return x


def _get_modality_encoder_module(name, modality_infos, key):
    """Instantiate encoder module for one modality.

    Inputs:
        name (str): Modality name.
        modality_infos (ModalityInfos): Modality metadata map.
        key (PRNGKeyArray): Random key for module initialization.

    Outputs:
        eqx.Module: Encoder module matching modality type.
    """
    modality = modality_infos[name]
    if modality.modality_type == "image":
        module = PatchEmbedding(
            input_shape=modality.input_shape,
            output_shape=modality.embedding_dim, 
            patch_size=modality.patch_size,
            embedding_dim=modality.embedding_dim,
            num_patches=modality.sequence_length,
            key=key
        )
    elif modality.modality_type == "vector":
        dim = prod(modality.input_shape)
        module = ParameterEmbedding(dim, modality.embedding_dim, key=key)
    elif modality.modality_type == "set":
        module = SetEmbedding(modality.input_shape[1], modality.embedding_dim, key=key)
    else:
        raise ValueError(f"Unsupported modality type: {modality.modality_type}")
    return module
    

def _patchify(x, *, image_shape, patch_size):
    """Split image-like tensors into flattened patch tokens.

    Inputs:
        x (Array): Input tensor in ``(C, L)`` or ``(C, H, W)`` layout.
        image_shape (Sequence[int]): Expected input shape.
        patch_size (int): Spatial patch size.

    Outputs:
        Array: Patch matrix with shape ``(S, patch_dim)``.
    """
    if len(image_shape) == 3:
        c, H, W = image_shape
        ph = pw = patch_size
        assert x.shape == (c, H, W), (x.shape, (c, H, W))
        return rearrange(
            x,
            "c (h ph) (w pw) -> (h w) (c ph pw)",
            ph=ph,
            pw=pw,
        )
    if len(image_shape) == 2:
        c, L = image_shape
        p = patch_size
        assert x.shape == (c, L), (x.shape, (c, L))
        return rearrange(
            x,
            "c (l p) -> l (c p)",
            p=p,
        )
    raise ValueError("patchify expects image_shape as (C, L) or (C, H, W).")


def get_patchify(image_shape, patch_size):
    """Create a ``patchify`` function bound to shape and patch size."""
    return partial(_patchify, image_shape=image_shape, patch_size=patch_size)


def _unpatchify(x, *, image_shape, patch_size):
    """Reassemble flattened patch vectors into image-like tensors.

    Inputs:
        x (Array): Patch matrix with shape ``(S, patch_dim)``.
        image_shape (Sequence[int]): Target shape ``(C, L)`` or ``(C, H, W)``.
        patch_size (int): Spatial patch size used in tokenization.

    Outputs:
        Array: Reconstructed tensor in ``image_shape`` layout.
    """
    if len(image_shape) == 3:
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
    if len(image_shape) == 2:
        c, L = image_shape
        p = patch_size
        l = L // p
        assert x.shape[0] == l, (x.shape, l)
        assert x.shape[1] == c * p, (x.shape, c, p)
        return rearrange(
            x,
            "l (c p) -> c (l p)",
            c=c,
            p=p,
        )
    raise ValueError("unpatchify expects image_shape as (C, L) or (C, H, W).")


def get_unpatchify(image_shape, patch_size):
    """Create an ``unpatchify`` function bound to shape and patch size.

    Inputs:
        image_shape (Sequence[int]): Target shape ``(C, L)`` or ``(C, H, W)``.
        patch_size (int): Patch size used for tokenization.

    Outputs:
        Callable: Function mapping patch tokens back to image-like tensors.
    """
    return partial(_unpatchify, image_shape=image_shape, patch_size=patch_size)


def _get_modality_decoder_module(name, modality_infos, key):
    """Instantiate decoder module for one modality.

    Inputs:
        name (str): Modality name.
        modality_infos (ModalityInfos): Modality metadata map.
        key (PRNGKeyArray): Random key for module initialization.

    Outputs:
        eqx.Module: Decoder module matching modality type.
    """
    # Assign decoding modules to corresponding modalities 
    # - possibly assign by 'modality_infos.modality_type == "image"' or something
    modality = modality_infos[name]
    if modality.modality_type == "image":
        n_channels = modality_infos[name].input_shape[0]
        spatial_ndim = len(modality.input_shape) - 1
        dim = modality.patch_size ** spatial_ndim * n_channels
        module = ImageDecoder(
            modality.embedding_dim, 
            dim, 
            image_shape=modality.input_shape, 
            patch_size=modality.patch_size, 
            key=key
        )
    elif modality.modality_type == "vector":
        dim = prod(modality.input_shape)
        module = ParameterDecoder(modality.embedding_dim, dim, key=key)
    elif modality.modality_type == "set":
        module = SetDecoder(modality.embedding_dim, modality.input_shape[1], key=key)
    else:
        raise ValueError(f"Unsupported modality type: {modality.modality_type}")
    return module


def _get_encoders(encoders, keys):
    """Select encoder submodules for a given modality key order.

    Inputs:
        encoders (dict[str, eqx.Module]): All modality encoders.
        keys (Iterable[str]): Requested modality ordering.

    Outputs:
        dict[str, eqx.Module]: Encoder mapping restricted to ``keys`` order.
    """
    return dict(zip(keys, [encoders[key] for key in keys]))


def _get_decoders(decoders, keys):
    """Select decoder submodules for a given modality key order.

    Inputs:
        decoders (dict[str, eqx.Module]): All modality decoders.
        keys (Iterable[str]): Requested modality ordering.

    Outputs:
        dict[str, eqx.Module]: Decoder mapping restricted to ``keys`` order.
    """
    return dict(zip(keys, [decoders[key] for key in keys]))


class ModalitiesEncoder(eqx.Module):
    """Encode a dictionary of modalities into token embeddings.

    Inputs:
        modality_infos (ModalityInfos): Modality metadata definitions.
        key (PRNGKeyArray): Random key for module initialization.

    Outputs:
        dict[str, Array] | Array: Per-modality embeddings or one concatenated sequence.
    """
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

    def __call__(
        self, 
        d: dict[str, Float[Array, "..."]], 
        stacked: bool = False
    ) -> dict[str, Float[Array, "..."]] | Float[Array, "..."]:
        """Encode modalities to token space.

        Inputs:
            d (dict[str, Array]): Modalities keyed by name.
            stacked (bool): If ``True``, concatenate all modality tokens.

        Outputs:
            dict[str, Array] | Array: Encoded tokens per modality or as one sequence.
        """
        e = jax.tree.map(
            lambda d, e: jnp.atleast_2d(e(d)), 
            d, 
            _get_encoders(self.encoders, keys=d.keys())
        )
        if stacked:
            e = jnp.concatenate(jax.tree.leaves(e))
        return e


class ModalitiesDecoder(eqx.Module):
    """Decode per-modality token embeddings back to data space.

    Inputs:
        modality_infos (ModalityInfos): Modality metadata definitions.
        key (PRNGKeyArray): Random key for module initialization.

    Outputs:
        dict[str, Array]: Decoded modalities keyed by name.
    """
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

    def __call__(
        self, 
        e: dict[str, Float[Array, "..."]]
    ) -> dict[str, Float[Array, "..."]]:
        """Decode token embeddings into modality tensors.

        Inputs:
            e (dict[str, Array]): Per-modality token sequences.

        Outputs:
            dict[str, Array]: Decoded modality tensors.
        """
        d = jax.tree.map(
            lambda e, d: d(e),
            e, 
            _get_decoders(self.decoders, keys=e.keys())
        )
        return d
