from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, Scalar

from ..naming import Modalities

LossFn = Callable[[eqx.Module, Modalities, PRNGKeyArray], tuple[Scalar, dict[str, Any]]]

SampleFn = Callable[[eqx.Module, Modalities, PRNGKeyArray], Modalities]

ConditionalSampleFn = Callable[[eqx.Module, Modalities, Sequence[str], PRNGKeyArray], Modalities]


@dataclass(frozen=True)
class GenerativeFamily:
    """
        Container for a generative model and its family-specific entry points
        for sampling and training loss.
    """

    name: str
    model: eqx.Module
    loss_fn: LossFn
    sample_fn: SampleFn
    sample_modalities_fn: Optional[ConditionalSampleFn] = None
