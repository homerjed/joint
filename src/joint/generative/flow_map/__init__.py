import jax.random as jr

from ...modules import ModalitiesDecoder, ModalitiesEncoder
from ...multimodal import MultiModalDiT, MultimodalTokenizer
from ...transformer import VisionTransformer
from ..base import GenerativeFamily
from ..flow_matching.time import get_time_embedder
from .loss import build_loss_fn, distillation_residual
from .model import FlowMapModel
from .sampling import _time_grid, single_sample_fn, single_sample_modalities_fn


def build_family(config, modality_infos, key):
    """Build the Boffi et al. (2025) self-distilled flow-map family."""
    key_encoder, key_decoder, key_transformer, key_time, key_interval = jr.split(key, 5)
    backbone_config = config.model_config.backbone
    family_config = config.model_config.family
    tokenizer = MultimodalTokenizer(
        modalities_encoder=ModalitiesEncoder(modality_infos, key=key_encoder),
        modalities_decoder=ModalitiesDecoder(modality_infos, key=key_decoder),
        modality_infos=modality_infos,
    )
    transformer = VisionTransformer(
        embedding_dim=backbone_config.embedding_dim,
        hidden_dim=backbone_config.hidden_dim,
        num_heads=backbone_config.n_heads,
        num_layers=family_config.n_blocks,
        dropout_rate=backbone_config.dropout_rate,
        key=key_transformer,
    )
    model = FlowMapModel(
        backbone=MultiModalDiT(tokenizer=tokenizer, transformer=transformer),
        time_embedder=get_time_embedder(
            backbone_config.embedding_dim, t_emb_dim=family_config.t_emb_dim, key=key_time
        ),
        interval_embedder=get_time_embedder(
            backbone_config.embedding_dim, t_emb_dim=family_config.t_emb_dim, key=key_interval
        ),
    )
    sampling_config = config.sampling_config
    n_steps = getattr(sampling_config, "flow_map_steps", 4)
    t0, t1 = float(sampling_config.t0), float(sampling_config.t1)
    _time_grid(n_steps, t0, t1, None)
    loss_fn = build_loss_fn(
        sampling_config.path_factory(), train_config=config.train_config, t0=t0, t1=t1
    )

    def sample_fn(model_instance, x_0, sample_key):
        return single_sample_fn(
            model_instance, x_0, n_steps=n_steps, t0=t0, t1=t1, key=sample_key
        )

    def sample_modalities_fn(model_instance, x_0, modalities_to_sample, sample_key):
        return single_sample_modalities_fn(
            model_instance, x_0, modalities_to_sample,
            n_steps=n_steps, t0=t0, t1=t1, key=sample_key,
        )

    return GenerativeFamily(
        name="flow_map", model=model, loss_fn=loss_fn, sample_fn=sample_fn,
        sample_modalities_fn=sample_modalities_fn,
    )


__all__ = [
    "FlowMapModel", "build_family", "build_loss_fn", "distillation_residual",
    "single_sample_fn", "single_sample_modalities_fn",
]
