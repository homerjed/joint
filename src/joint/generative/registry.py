from .flow_matching import build_family as build_flow_matching_family
from .flow_map import build_family as build_flow_map_family
from .sphere_encoder import build_family as build_sphere_encoder_family


FAMILY_BUILDERS = {
    "flow_matching": build_flow_matching_family,
    "flow_map": build_flow_map_family,
    "sphere_encoder": build_sphere_encoder_family,
}


def build_model_family(config, modality_infos, key):
    family_name = str(config.model_config.model_family)
    try:
        builder = FAMILY_BUILDERS[family_name]
    except KeyError as err:
        raise ValueError(
            f"Unknown model_family {family_name!r}. Expected one of {sorted(FAMILY_BUILDERS)}."
        ) from err
    return builder(config, modality_infos, key)
