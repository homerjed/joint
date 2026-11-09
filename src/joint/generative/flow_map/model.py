from typing import Optional, Sequence

import equinox as eqx
import jax
from jaxtyping import Array, Bool, PRNGKeyArray, Scalar

from ...multimodal import MultiModalDiT
from ...naming import Modalities


class FlowMapModel(eqx.Module):
    """Two-time average velocity, with X(s,t,x) = x + (t-s) v(s,t,x).

    The diagonal v(s,s,x) is the instantaneous velocity. Separate embeddings
    of the start time and duration let the shared DiT represent both objects.
    See Boffi et al. (2025), https://arxiv.org/abs/2505.18825.
    """

    backbone: MultiModalDiT
    time_embedder: eqx.Module
    interval_embedder: eqx.Module

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
        s: Scalar,
        t: Scalar,
        *,
        masks=None,
        attention_mask=None,
        key: Optional[PRNGKeyArray] = None,
        use_class_token: bool = False,
    ):
        """Evaluate v(s,t,x); this returns a velocity, not the transported state."""
        condition = self.time_embedder(s) + self.interval_embedder(t - s)
        return self.backbone(
            modalities,
            condition=condition,
            masks=masks,
            attention_mask=attention_mask,
            key=key,
            use_class_token=use_class_token,
        )

    def flow_map(
        self,
        modalities: Modalities,
        s: Scalar,
        t: Scalar,
        *,
        masks=None,
        attention_mask=None,
        key: Optional[PRNGKeyArray] = None,
    ) -> Modalities:
        """Transport in one call; X(s,s,x) is x on valid entries, with padding zero."""
        if masks is not None:
            modalities = self.backbone.tokenizer.mask_modalities(modalities, masks)
        velocity = self(
            modalities, s, t, masks=masks, attention_mask=attention_mask, key=key
        )
        result = jax.tree.map(lambda x, v: x + (t - s) * v, modalities, velocity)
        if masks is not None:
            result = self.backbone.tokenizer.mask_modalities(result, masks)
        return result
