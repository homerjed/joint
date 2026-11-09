from itertools import accumulate
from typing import Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from .modules import ModalitiesDecoder, ModalitiesEncoder
from .naming import Modalities
from .transformer import VisionTransformer
from .utils import ModalityInfos


class MultimodalTokenizer(eqx.Module):
    """Shared multimodal tokenization and detokenization utilities."""

    modalities_encoder: ModalitiesEncoder
    modalities_decoder: ModalitiesDecoder
    modality_infos: ModalityInfos

    def canonical_keys(
        self,
        modalities: Optional[Modalities] = None,
        *,
        keys: Optional[Sequence[str]] = None,
    ) -> tuple[str, ...]:
        if keys is not None:
            return tuple(keys)
        if modalities is not None:
            return tuple(sorted(modalities.keys()))
        return tuple(sorted(self.modality_infos.keys()))

    def normalize_modalities(
        self,
        modalities: Modalities,
        *,
        keys: Optional[Sequence[str]] = None,
    ) -> Modalities:
        ordered_keys = self.canonical_keys(modalities, keys=keys)
        return {name: modalities[name] for name in ordered_keys}

    def get_modality_lengths(self, keys: Optional[Sequence[str]] = None) -> list[int]:
        ordered_keys = self.canonical_keys(keys=keys)
        return [self.modality_infos[name].sequence_length for name in ordered_keys]

    def get_split_indices(self, keys: Optional[Sequence[str]] = None) -> list[int]:
        lengths = self.get_modality_lengths(keys)[:-1]
        return list(accumulate(lengths))

    def total_sequence_length(self, keys: Optional[Sequence[str]] = None) -> int:
        return sum(self.get_modality_lengths(keys))

    def get_modality_token_owners(
        self,
        keys: Optional[Sequence[str]] = None,
    ) -> Int[Array, "s"]:
        lengths = self.get_modality_lengths(keys)
        owners = [
            jnp.full((length,), idx, dtype=jnp.int32)
            for idx, length in enumerate(lengths)
        ]
        return jnp.concatenate(owners)

    def build_directional_attention_mask(
        self,
        conditioning_mask: Bool[Array, "m"],
        keys: Optional[Sequence[str]] = None,
    ) -> Bool[Array, "s s"]:
        token_owner = self.get_modality_token_owners(keys)
        query_is_conditioning = conditioning_mask[token_owner][:, None]
        key_is_conditioning = conditioning_mask[token_owner][None, :]
        return jnp.logical_or(
            key_is_conditioning,
            jnp.logical_not(query_is_conditioning),
        )

    def build_attention_mask_from_modal_masks(
        self,
        masks: dict[str, Bool[Array, "..."]],
        *,
        keys: Optional[Sequence[str]] = None,
    ) -> Bool[Array, "s s"]:
        token_mask = self.token_validity_mask(masks, keys=keys)
        return jnp.broadcast_to(
            token_mask[jnp.newaxis, :],
            (token_mask.shape[0], token_mask.shape[0]),
        )

    def _modality_token_mask(self, name: str, mask) -> Bool[Array, "s"]:
        valid = jnp.asarray(mask, dtype=jnp.bool_).reshape(-1)
        expected = self.modality_infos[name].sequence_length
        if valid.shape != (expected,):
            raise ValueError(f"Mask for {name!r} must contain {expected} token flags.")
        return valid

    def token_validity_mask(
        self,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
        *,
        keys: Optional[Sequence[str]] = None,
    ) -> Bool[Array, "s"]:
        """Stack token-validity flags; omitted modalities are entirely valid."""
        masks = {} if masks is None else masks
        unknown = set(masks) - set(self.modality_infos)
        if unknown:
            raise ValueError(f"Masks supplied for unknown modalities: {sorted(unknown)}")
        return jnp.concatenate([
            self._modality_token_mask(name, masks[name])
            if name in masks
            else jnp.ones((self.modality_infos[name].sequence_length,), dtype=jnp.bool_)
            for name in self.canonical_keys(keys=keys)
        ])

    def raw_validity_mask(self, name: str, mask) -> Bool[Array, "..."]:
        """Expand one example's token flags to the modality's raw input shape."""
        spec = self.modality_infos[name]
        valid = self._modality_token_mask(name, mask)
        if spec.modality_type == "set":
            valid = valid[:, None]
        elif spec.modality_type == "vector":
            valid = valid.reshape(())
        elif spec.modality_type == "image":
            grid_shape = tuple(size // spec.patch_size for size in spec.input_shape[1:])
            valid = valid.reshape(grid_shape)
            for axis in range(len(grid_shape)):
                valid = jnp.repeat(valid, spec.patch_size, axis=axis)
            valid = valid[None, ...]
        else:
            raise ValueError(f"Unsupported modality type: {spec.modality_type}")
        return jnp.broadcast_to(valid, spec.input_shape)

    def mask_modalities(
        self,
        modalities: Modalities,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
    ) -> Modalities:
        """Clear invalid raw entries before arithmetic, including padded NaNs."""
        if masks is None:
            return modalities
        return {
            name: jnp.where(self.raw_validity_mask(name, masks[name]), value, 0)
            if name in masks else value
            for name, value in modalities.items()
        }

    def encode(
        self,
        modalities: Modalities,
        *,
        keys: Optional[Sequence[str]] = None,
        stacked: bool = False,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
    ) -> Modalities | Float[Array, "s e"]:
        ordered_modalities = self.normalize_modalities(modalities, keys=keys)
        ordered_modalities = self.mask_modalities(ordered_modalities, masks)
        tokens = self.modalities_encoder(ordered_modalities)
        if masks is not None:
            tokens = {
                name: jnp.where(self._modality_token_mask(name, masks[name])[:, None], value, 0)
                if name in masks else value
                for name, value in tokens.items()
            }
        if stacked:
            return self.stack_tokens(tokens, keys=ordered_modalities.keys())
        return tokens

    def stack_tokens(
        self,
        tokens: Modalities,
        *,
        keys: Optional[Sequence[str]] = None,
    ) -> Float[Array, "s e"]:
        ordered_tokens = self.normalize_modalities(tokens, keys=keys)
        return jnp.concatenate(jax.tree.leaves(ordered_tokens))

    def unstack_tokens(
        self,
        tokens: Float[Array, "s e"],
        *,
        keys: Optional[Sequence[str]] = None,
    ) -> Modalities:
        ordered_keys = self.canonical_keys(keys=keys)
        split_indices = self.get_split_indices(ordered_keys)
        if len(ordered_keys) == 1:
            token_chunks = [tokens]
        else:
            token_chunks = jnp.split(tokens, split_indices)
        return {name: chunk for name, chunk in zip(ordered_keys, token_chunks)}

    def decode(
        self,
        tokens: Modalities | Float[Array, "s e"],
        *,
        keys: Optional[Sequence[str]] = None,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
    ) -> Modalities:
        if not isinstance(tokens, dict):
            tokens = self.unstack_tokens(tokens, keys=keys)
        ordered_tokens = self.normalize_modalities(tokens, keys=keys)
        if masks is not None:
            ordered_tokens = {
                name: jnp.where(self._modality_token_mask(name, masks[name])[:, None], value, 0)
                if name in masks else value
                for name, value in ordered_tokens.items()
            }
        return self.mask_modalities(self.modalities_decoder(ordered_tokens), masks)


class MultiModalDiT(eqx.Module):
    """Generic multimodal DiT backbone over shared tokenization utilities."""

    tokenizer: MultimodalTokenizer
    transformer: VisionTransformer

    def _resolve_attention_mask(
        self,
        modalities: Modalities,
        *,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
        attention_mask: Optional[Bool[Array, "s s"]] = None,
    ) -> Optional[Bool[Array, "s s"]]:
        keys = self.tokenizer.canonical_keys(modalities)
        mask = attention_mask
        if mask is not None:
            length = self.tokenizer.total_sequence_length(keys)
            mask = jnp.asarray(mask, dtype=jnp.bool_)
            if mask.shape != (length, length):
                raise ValueError(f"attention_mask must have shape {(length, length)}.")
        if masks is not None:
            valid_keys = self.tokenizer.build_attention_mask_from_modal_masks(masks, keys=keys)
            mask = valid_keys if mask is None else mask & valid_keys
        return mask

    def hidden_tokens(
        self,
        modalities: Modalities,
        *,
        condition: Optional[Float[Array, "e"]] = None,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
        attention_mask: Optional[Bool[Array, "s s"]] = None,
        key: Optional[PRNGKeyArray] = None,
        use_class_token: bool = False,
    ) -> Float[Array, "s e"] | Float[Array, "1 e"]:
        ordered_keys = self.tokenizer.canonical_keys(modalities)
        mask = self._resolve_attention_mask(
            modalities,
            masks=masks,
            attention_mask=attention_mask,
        )
        token_mask = (
            self.tokenizer.token_validity_mask(masks, keys=ordered_keys)
            if masks is not None else None
        )
        tokens = self.tokenizer.encode(modalities, keys=ordered_keys, stacked=True, masks=masks)
        hidden = self.transformer(
            tokens,
            condition,
            mask=mask,
            key=key,
            use_class_token=use_class_token,
            token_mask=token_mask,
        )
        return hidden

    def decode_hidden(
        self,
        hidden_tokens: Float[Array, "s e"],
        *,
        keys: Optional[Sequence[str]] = None,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
    ) -> Modalities:
        decoded = self.tokenizer.decode(hidden_tokens, keys=keys, masks=masks)
        return decoded

    def build_directional_attention_mask(
        self,
        conditioning_mask: Bool[Array, "m"],
        keys: Optional[Sequence[str]] = None,
    ) -> Bool[Array, "s s"]:
        mask = self.tokenizer.build_directional_attention_mask(conditioning_mask, keys)
        return mask

    def __call__(
        self,
        modalities: Modalities,
        *,
        condition: Optional[Float[Array, "e"]] = None,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
        attention_mask: Optional[Bool[Array, "s s"]] = None,
        key: Optional[PRNGKeyArray] = None,
        use_class_token: bool = False,
    ) -> Modalities | Float[Array, "1 e"]:
        ordered_keys = self.tokenizer.canonical_keys(modalities)
        hidden = self.hidden_tokens(
            modalities,
            condition=condition,
            masks=masks,
            attention_mask=attention_mask,
            key=key,
            use_class_token=use_class_token,
        )
        if use_class_token:
            return hidden
        decoded = self.decode_hidden(hidden, keys=ordered_keys, masks=masks)
        return decoded
