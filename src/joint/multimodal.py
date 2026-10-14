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
        ordered_masks = self.normalize_modalities(masks, keys=keys)
        token_mask = jnp.concatenate([jnp.ravel(ordered_masks[name]) for name in ordered_masks])
        token_mask = jnp.asarray(token_mask, dtype=jnp.bool_)
        return jnp.broadcast_to(
            token_mask[jnp.newaxis, :],
            (token_mask.shape[0], token_mask.shape[0]),
        )

    def encode(
        self,
        modalities: Modalities,
        *,
        keys: Optional[Sequence[str]] = None,
        stacked: bool = False,
    ) -> Modalities | Float[Array, "s e"]:
        ordered_modalities = self.normalize_modalities(modalities, keys=keys)
        tokens = self.modalities_encoder(ordered_modalities)
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
    ) -> Modalities:
        if not isinstance(tokens, dict):
            tokens = self.unstack_tokens(tokens, keys=keys)
        ordered_tokens = self.normalize_modalities(tokens, keys=keys)
        return self.modalities_decoder(ordered_tokens)


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
        if masks is not None and attention_mask is not None:
            raise ValueError("Specify either masks or attention_mask, not both.")

        if attention_mask is not None:
            return attention_mask
        if masks is None:
            return None
        mask = self.tokenizer.build_attention_mask_from_modal_masks(
            masks,
            keys=self.tokenizer.canonical_keys(modalities),
        )
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
        tokens = self.tokenizer.encode(modalities, keys=ordered_keys, stacked=True)
        hidden = self.transformer(
            tokens,
            condition,
            mask=mask,
            key=key,
            use_class_token=use_class_token,
        )
        return hidden

    def decode_hidden(
        self,
        hidden_tokens: Float[Array, "s e"],
        *,
        keys: Optional[Sequence[str]] = None,
    ) -> Modalities:
        decoded = self.tokenizer.decode(hidden_tokens, keys=keys)
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
        decoded = self.decode_hidden(hidden, keys=ordered_keys)
        return decoded
