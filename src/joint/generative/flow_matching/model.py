from typing import Callable, Optional, Sequence

import equinox as eqx
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar

from ...multimodal import MultiModalDiT
from ...naming import Modalities


class FlowMatchingModel(eqx.Module):
    """Flow-matching model built on a shared multimodal DiT backbone."""

    backbone: MultiModalDiT
    time_embedder: eqx.Module | Callable[[Scalar], Float[Array, "e"]]

    def get_modality_lengths(self, keys: Optional[Sequence[str]] = None) -> list[int]:
        return self.backbone.tokenizer.get_modality_lengths(keys)

    def build_directional_attention_mask(
        self,
        conditioning_mask: Bool[Array, "m"],
        keys: Optional[Sequence[str]] = None,
    ) -> Bool[Array, "s s"]:
        return self.backbone.build_directional_attention_mask(conditioning_mask, keys)

    def __call__(
        self,
        modalities: Modalities,
        t: Optional[Scalar] = None,
        masks: Optional[dict[str, Bool[Array, "..."]]] = None,
        attention_mask: Optional[Bool[Array, "s s"]] = None,
        key: Optional[PRNGKeyArray] = None,
        *,
        use_class_token: bool = False,
    ) -> Modalities | Float[Array, "1 e"]:
        condition = None if t is None else self.time_embedder(t)
        return self.backbone(
            modalities,
            condition=condition,
            masks=masks,
            attention_mask=attention_mask,
            key=key,
            use_class_token=use_class_token,
        )
