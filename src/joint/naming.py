from jaxtyping import Array

from .utils import ModalitySpec, ModalityInfos

# Is there a way to dynamically type the modality dicts or mask dicts?
Modalities = dict[str, Array]