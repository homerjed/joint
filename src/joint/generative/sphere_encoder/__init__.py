import equinox as eqx
import jax.random as jr

from ...modules import ModalitiesDecoder, ModalitiesEncoder
from ...multimodal import MultimodalTokenizer
from ...transformer import VisionTransformer
from ..base import GenerativeFamily
from .loss import build_loss_fn
from .model import SphereEncoderModel
from .sampling import single_sample_fn, single_sample_modalities_fn


def build_family(config, modality_infos, key):
    key_encoder, key_decoder, key_encoder_transformer, key_decoder_transformer, key_latent_in, key_latent_out = jr.split(
        key,
        6,
    )
    backbone_config = config.model_config.backbone
    family_config = config.model_config.family

    modalities_encoder = ModalitiesEncoder(modality_infos, key=key_encoder)
    modalities_decoder = ModalitiesDecoder(modality_infos, key=key_decoder)
    tokenizer = MultimodalTokenizer(
        modalities_encoder=modalities_encoder,
        modalities_decoder=modalities_decoder,
        modality_infos=modality_infos,
    )
    encoder_transformer = VisionTransformer(
        embedding_dim=backbone_config.embedding_dim,
        hidden_dim=backbone_config.hidden_dim,
        num_heads=backbone_config.n_heads,
        num_layers=family_config.n_blocks_encoder,
        dropout_rate=backbone_config.dropout_rate,
        key=key_encoder_transformer,
    )
    decoder_transformer = VisionTransformer(
        embedding_dim=backbone_config.embedding_dim,
        hidden_dim=backbone_config.hidden_dim,
        num_heads=backbone_config.n_heads,
        num_layers=family_config.n_blocks_decoder,
        dropout_rate=backbone_config.dropout_rate,
        key=key_decoder_transformer,
    )
    latent_dim = int(getattr(family_config, "latent_dim", backbone_config.embedding_dim))
    model = SphereEncoderModel(
        tokenizer=tokenizer,
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
        latent_in=eqx.nn.Linear(
            backbone_config.embedding_dim,
            latent_dim,
            key=key_latent_in,
        ),
        latent_out=eqx.nn.Linear(
            latent_dim,
            backbone_config.embedding_dim,
            key=key_latent_out,
        ),
        latent_dim=latent_dim,
        noise_sigma_max_angle=float(getattr(family_config, "noise_sigma_max_angle", 75.0)),
    )

    loss_fn = build_loss_fn(config.train_config, modality_infos)
    forward_steps = int(getattr(config.sampling_config, "forward_steps", 1))
    use_sampling_scheduler = bool(
        getattr(config.sampling_config, "use_sampling_scheduler", False)
    )
    cache_sampling_noise = bool(
        getattr(config.sampling_config, "cache_sampling_noise", False)
    )

    def sample_fn(model_instance, x_0, sample_key):
        return single_sample_fn(
            model_instance,
            x_0,
            forward_steps=forward_steps,
            use_sampling_scheduler=use_sampling_scheduler,
            cache_sampling_noise=cache_sampling_noise,
            key=sample_key,
        )

    def sample_modalities_fn(model_instance, x_0, modalities_to_sample, sample_key):
        return single_sample_modalities_fn(
            model_instance,
            x_0,
            list(modalities_to_sample),
            forward_steps=forward_steps,
            use_sampling_scheduler=use_sampling_scheduler,
            cache_sampling_noise=cache_sampling_noise,
            key=sample_key,
        )

    return GenerativeFamily(
        name="sphere_encoder",
        model=model,
        loss_fn=loss_fn,
        sample_fn=sample_fn,
        sample_modalities_fn=sample_modalities_fn,
    )
