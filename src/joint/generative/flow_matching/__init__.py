import jax.random as jr

from ...modules import ModalitiesDecoder, ModalitiesEncoder
from ...multimodal import MultiModalDiT, MultimodalTokenizer
from ...transformer import VisionTransformer
from ..base import GenerativeFamily
from .loss import build_loss_fn
from .model import FlowMatchingModel
from .path import OptimalTransportPath, clip_time
from .sampling import single_sample_fn_ode, single_sample_modalities_fn_ode
from .solvers import SolutionKwargs
from .time import TimeEmbeddingMLP, get_time_embedder, get_timestep_embedding


def build_family(config, modality_infos, key):
    key_encoder, key_decoder, key_transformer, key_time = jr.split(key, 4)
    backbone_config = config.model_config.backbone
    family_config = config.model_config.family

    modalities_encoder = ModalitiesEncoder(modality_infos, key=key_encoder)
    modalities_decoder = ModalitiesDecoder(modality_infos, key=key_decoder)
    tokenizer = MultimodalTokenizer(
        modalities_encoder=modalities_encoder,
        modalities_decoder=modalities_decoder,
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
    backbone = MultiModalDiT(tokenizer=tokenizer, transformer=transformer)
    time_embedder = get_time_embedder(
        backbone_config.embedding_dim,
        t_emb_dim=int(getattr(family_config, "t_emb_dim", backbone_config.embedding_dim)),
        key=key_time,
    )
    model = FlowMatchingModel(backbone=backbone, time_embedder=time_embedder)

    path = config.sampling_config.path_factory()
    if not isinstance(path, OptimalTransportPath):
        path = config.sampling_config.path_factory()
    solution_kwargs = SolutionKwargs(
        config.sampling_config.solver_factory(),
        config.sampling_config.t0,
        config.sampling_config.t1,
        config.sampling_config.dt0,
    )
    loss_fn = build_loss_fn(
        path,
        solution_kwargs=solution_kwargs,
        train_config=config.train_config,
    )

    def sample_fn(model_instance, x_0, sample_key):
        return single_sample_fn_ode(
            model_instance,
            x_0,
            solution_kwargs=solution_kwargs,
            key=sample_key,
        )

    def sample_modalities_fn(model_instance, x_0, modalities_to_sample, sample_key):
        return single_sample_modalities_fn_ode(
            model_instance,
            x_0,
            list(modalities_to_sample),
            solution_kwargs=solution_kwargs,
            key=sample_key,
            bridge_path=path,
        )

    return GenerativeFamily(
        name="flow_matching",
        model=model,
        loss_fn=loss_fn,
        sample_fn=sample_fn,
        sample_modalities_fn=sample_modalities_fn,
    )


__all__ = [
    "FlowMatchingModel",
    "GenerativeFamily",
    "OptimalTransportPath",
    "SolutionKwargs",
    "TimeEmbeddingMLP",
    "build_family",
    "build_loss_fn",
    "clip_time",
    "get_time_embedder",
    "get_timestep_embedding",
    "single_sample_fn_ode",
    "single_sample_modalities_fn_ode",
]
