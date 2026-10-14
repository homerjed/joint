"""Naming aliases shared across multimodal modules.

Inputs:
    Modality dictionaries map string names to JAX arrays.

Outputs:
    ``Modalities`` provides a concise type alias for these mappings.
"""

from typing import Callable
from jaxtyping import Array, Scalar, Float

from .utils import ModalitySpec, ModalityInfos

Modalities = dict[str, Array]

TimeScalar = Scalar | float

TimeEmbedder = Callable[[TimeScalar], Float[Array, "e"]]