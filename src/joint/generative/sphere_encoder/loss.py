from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ...utils import ModalityInfos
from .model import SphereEncoderModel


def reconstruction_loss(prediction: jnp.ndarray, target: jnp.ndarray, kind: str) -> jnp.ndarray:
    delta = prediction - target
    if kind == "l1":
        return jnp.mean(jnp.abs(delta))
    if kind == "l2":
        return jnp.mean(jnp.square(delta))
    if kind in {"l1+l2", "l2+l1"}:
        return 0.5 * (jnp.mean(jnp.abs(delta)) + jnp.mean(jnp.square(delta)))
    raise ValueError(f"Unsupported reconstruction loss: {kind}")


def cosine_distance(x: jnp.ndarray, y: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = x.reshape((x.shape[0], -1))
    y = y.reshape((y.shape[0], -1))
    numerator = jnp.sum(x * y, axis=-1)
    denominator = jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1)
    return 1.0 - numerator / jnp.maximum(denominator, eps)


def _depthwise_conv(images: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    channels = images.shape[1]
    kernel = jnp.asarray(kernel, dtype=images.dtype)
    kernel = jnp.tile(kernel[None, None, :, :], (channels, 1, 1, 1))
    return jax.lax.conv_general_dilated(
        images,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        feature_group_count=channels,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )


def _avg_pool2x2(images: jnp.ndarray) -> jnp.ndarray:
    return (
        jax.lax.reduce_window(
            images,
            0.0,
            jax.lax.add,
            window_dimensions=(1, 1, 2, 2),
            window_strides=(1, 1, 2, 2),
            padding="SAME",
        )
        / 4.0
    )


def _edge_magnitude(images: jnp.ndarray) -> jnp.ndarray:
    sobel_x = jnp.array(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        dtype=images.dtype,
    ) / 4.0
    sobel_y = sobel_x.T
    grad_x = _depthwise_conv(images, sobel_x)
    grad_y = _depthwise_conv(images, sobel_y)
    return jnp.sqrt(jnp.square(grad_x) + jnp.square(grad_y) + 1e-6)


def perceptual_loss(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    num_scales: int = 3,
) -> jnp.ndarray:
    gaussian = jnp.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=prediction.dtype,
    ) / 16.0

    pred = prediction.astype(jnp.float32)
    tgt = target.astype(jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    weight = jnp.asarray(1.0, dtype=jnp.float32)
    weight_sum = jnp.asarray(0.0, dtype=jnp.float32)

    for _ in range(num_scales):
        pred_blur = _depthwise_conv(pred, gaussian)
        tgt_blur = _depthwise_conv(tgt, gaussian)
        pred_edges = _edge_magnitude(pred_blur)
        tgt_edges = _edge_magnitude(tgt_blur)

        blur_term = jnp.mean(jnp.abs(pred_blur - tgt_blur))
        edge_term = jnp.mean(jnp.abs(pred_edges - tgt_edges))
        total = total + weight * (blur_term + 0.5 * edge_term)
        weight_sum = weight_sum + weight

        pred = _avg_pool2x2(pred_blur)
        tgt = _avg_pool2x2(tgt_blur)
        weight = weight * 0.5

    return total / jnp.maximum(weight_sum, 1e-6)


def stratified_unit_radii(
    key: jax.Array,
    size: tuple[int, ...],
    *,
    including_zero: bool = True,
    shuffle: bool = True,
) -> jax.Array:
    batch_size = int(size[0])
    tail_shape = tuple(size[1:])
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if batch_size == 1:
        values = jnp.array([0.0 if including_zero else 1.0], dtype=jnp.float32)
    else:
        body = batch_size - 2 if including_zero else batch_size - 1
        key_values, key_perm = jr.split(key)
        if body > 0:
            indices = jnp.arange(body, dtype=jnp.float32)
            offsets = jnp.clip(jr.uniform(key_values, (body,)), a_min=1e-3, a_max=1.0)
            values = (indices + offsets) / body
        else:
            values = jnp.empty((0,), dtype=jnp.float32)
        anchors = [1.0, 0.0] if including_zero else [1.0]
        values = jnp.concatenate([values, jnp.array(anchors, dtype=jnp.float32)])
        if shuffle:
            values = jr.permutation(key_perm, values)

    return values.reshape((batch_size,) + (1,) * len(tail_shape))


def _sample_conditioning_mask(
    key: PRNGKeyArray,
    n_modalities: int,
    *,
    conditional_training_probability: float,
) -> Optional[Bool[Array, "m"]]:
    if conditional_training_probability <= 0.0 or n_modalities < 2:
        return None

    key_use, key_perm, key_count = jr.split(key, 3)
    use_conditional = jr.bernoulli(key_use, conditional_training_probability)
    n_conditioning = jr.randint(key_count, (), minval=1, maxval=n_modalities)
    permutation = jr.permutation(key_perm, n_modalities)
    ranks = jnp.argsort(permutation)
    conditioning_mask = ranks < n_conditioning
    return jnp.where(use_conditional, conditioning_mask, jnp.zeros_like(conditioning_mask))


def _per_modality_loss_tree(prediction, target, *, kind: str):
    return {
        name: reconstruction_loss(prediction[name], target[name], kind)
        for name in sorted(target.keys())
    }


def _weighted_modality_total(per_modality_loss, generated_weights):
    modality_names = tuple(sorted(per_modality_loss.keys()))
    values = jnp.stack([per_modality_loss[name] for name in modality_names])
    if generated_weights is None:
        return jnp.sum(values), per_modality_loss

    loss_weights = generated_weights.astype(values.dtype)
    n_generated = jnp.maximum(jnp.sum(loss_weights), 1.0)
    total = jnp.sum(values * loss_weights) * (loss_weights.shape[0] / n_generated)
    weighted = {
        name: value
        for name, value in zip(modality_names, values)
    }
    return total, weighted


def _perceptual_tree(prediction, target, *, modality_infos: ModalityInfos):
    dtype = next(iter(target.values())).dtype
    per_modality = {}
    for name in sorted(target.keys()):
        spec = modality_infos[name]
        if spec.modality_type == "image" and len(spec.input_shape) == 3:
            per_modality[name] = perceptual_loss(prediction[name], target[name])
        else:
            per_modality[name] = jnp.asarray(0.0, dtype=dtype)
    return per_modality


def build_loss_fn(train_config, modality_infos: ModalityInfos):
    reconstruction_kind = str(getattr(train_config, "reconstruction_loss", "l1+l2"))
    reconstruction_perceptual_weight = float(
        getattr(train_config, "reconstruction_perceptual_weight", 0.0)
    )
    pixel_consistency_weight = float(
        getattr(train_config, "pixel_consistency_weight", 0.25)
    )
    pixel_consistency_perceptual_weight = float(
        getattr(train_config, "pixel_consistency_perceptual_weight", 0.0)
    )
    latent_consistency_weight = float(
        getattr(train_config, "latent_consistency_weight", 0.10)
    )
    mix_hard_cases = bool(getattr(train_config, "mix_hard_cases", True))
    mix_hard_cases_prob = float(getattr(train_config, "mix_hard_cases_prob", 0.10))
    mix_hard_cases_max_angle = float(
        getattr(
            train_config,
            "mix_hard_cases_max_angle",
            getattr(train_config, "noise_sigma_max_angle", 75.0),
        )
    )
    conditional_training_probability = float(
        getattr(train_config, "conditional_training_probability", 0.0)
    )

    def loss_fn(
        model: SphereEncoderModel,
        batch: dict[str, Float[Array, "n ..."]],
        key: PRNGKeyArray,
    ):
        n_batch = len(next(iter(batch.values())))
        (
            key_encode,
            key_conditioning,
            key_base_r,
            key_mix,
            key_heavy_noise,
            key_heavy_decode,
            key_light_r,
            key_light_noise,
            key_light_decode,
            key_reencode,
        ) = jr.split(key, 10)

        conditioning_mask = _sample_conditioning_mask(
            key_conditioning,
            n_modalities=len(batch),
            conditional_training_probability=conditional_training_probability,
        )
        generated_weights = None
        attention_mask = None
        if conditioning_mask is not None:
            attention_mask = model.build_directional_attention_mask(
                conditioning_mask,
                keys=sorted(batch.keys()),
            )
            generated_weights = jnp.logical_not(conditioning_mask)

        encode_keys = jr.split(key_encode, n_batch)
        raw_latents = jax.vmap(
            lambda d, sample_key: model.encode(
                d,
                attention_mask=attention_mask,
                key=sample_key,
            )
        )(batch, encode_keys)
        clean_latents = jax.lax.stop_gradient(raw_latents)
        sphere_latents = jax.vmap(model.spherify)(raw_latents)

        base_radii = stratified_unit_radii(key_base_r, sphere_latents.shape)
        heavy_radii = base_radii
        if mix_hard_cases:
            sigma = jnp.tan(jnp.deg2rad(model.noise_sigma_max_angle))
            hard_case_max = jnp.tan(jnp.deg2rad(mix_hard_cases_max_angle)) / sigma
            scalers = jr.uniform(key_mix, heavy_radii.shape, dtype=heavy_radii.dtype)
            mixed_radii = 1.0 + scalers * jnp.maximum(hard_case_max - 1.0, 0.0)
            keep_standard = (scalers > mix_hard_cases_prob).astype(heavy_radii.dtype)
            heavy_radii = keep_standard * heavy_radii + (1.0 - keep_standard) * mixed_radii

        heavy_noise_keys = jr.split(key_heavy_noise, n_batch)
        heavy_latents = jax.vmap(
            lambda sample_key, latents, radius: model.add_noise(sample_key, latents, radius)
        )(heavy_noise_keys, sphere_latents, heavy_radii)
        heavy_decode_keys = jr.split(key_heavy_decode, n_batch)
        heavy_recons = jax.vmap(
            lambda latents, sample_key: model.decode(
                latents,
                attention_mask=attention_mask,
                key=sample_key,
            )
        )(heavy_latents, heavy_decode_keys)

        if pixel_consistency_weight > 0.0 or pixel_consistency_perceptual_weight > 0.0:
            light_radii = 0.5 * base_radii * stratified_unit_radii(
                key_light_r,
                sphere_latents.shape,
            )
            light_noise_keys = jr.split(key_light_noise, n_batch)
            light_latents = jax.vmap(
                lambda sample_key, latents, radius: model.add_noise(sample_key, latents, radius)
            )(light_noise_keys, sphere_latents, light_radii)
            light_decode_keys = jr.split(key_light_decode, n_batch)
            light_recons = jax.vmap(
                lambda latents, sample_key: model.decode(
                    latents,
                    attention_mask=attention_mask,
                    key=sample_key,
                )
            )(light_latents, light_decode_keys)

            reconstruction_tree = _per_modality_loss_tree(
                light_recons,
                batch,
                kind=reconstruction_kind,
            )
            reconstruction_value, _ = _weighted_modality_total(
                reconstruction_tree,
                generated_weights,
            )

            reconstruction_perceptual_tree = _perceptual_tree(
                light_recons,
                batch,
                modality_infos=modality_infos,
            )
            reconstruction_perceptual_value, _ = _weighted_modality_total(
                reconstruction_perceptual_tree,
                generated_weights,
            )

            pixel_tree = _per_modality_loss_tree(
                heavy_recons,
                jax.tree.map(jax.lax.stop_gradient, light_recons),
                kind=reconstruction_kind,
            )
            pixel_value, per_modality_loss = _weighted_modality_total(
                pixel_tree,
                generated_weights,
            )

            pixel_perceptual_tree = _perceptual_tree(
                heavy_recons,
                jax.tree.map(jax.lax.stop_gradient, light_recons),
                modality_infos=modality_infos,
            )
            pixel_perceptual_value, _ = _weighted_modality_total(
                pixel_perceptual_tree,
                generated_weights,
            )
        else:
            reconstruction_tree = _per_modality_loss_tree(
                heavy_recons,
                batch,
                kind=reconstruction_kind,
            )
            reconstruction_value, per_modality_loss = _weighted_modality_total(
                reconstruction_tree,
                generated_weights,
            )
            reconstruction_perceptual_tree = _perceptual_tree(
                heavy_recons,
                batch,
                modality_infos=modality_infos,
            )
            reconstruction_perceptual_value, _ = _weighted_modality_total(
                reconstruction_perceptual_tree,
                generated_weights,
            )
            pixel_value = jnp.asarray(0.0, dtype=reconstruction_value.dtype)
            pixel_perceptual_value = jnp.asarray(0.0, dtype=reconstruction_value.dtype)

        if latent_consistency_weight > 0.0:
            reencode_keys = jr.split(key_reencode, n_batch)
            reencoded = jax.vmap(
                lambda modalities, sample_key: model.encode(
                    modalities,
                    attention_mask=attention_mask,
                    key=sample_key,
                )
            )(heavy_recons, reencode_keys)
            latent_value = jnp.mean(cosine_distance(reencoded, clean_latents))
        else:
            latent_value = jnp.asarray(0.0, dtype=reconstruction_value.dtype)

        total = (
            reconstruction_value
            + reconstruction_perceptual_weight * reconstruction_perceptual_value
            + pixel_consistency_weight * pixel_value
            + pixel_consistency_perceptual_weight * pixel_perceptual_value
            + latent_consistency_weight * latent_value
        )

        metrics = {
            "per_modality_loss": per_modality_loss,
            "reconstruction": reconstruction_value,
            "reconstruction_perceptual": reconstruction_perceptual_value,
            "pixel_consistency": pixel_value,
            "pixel_consistency_perceptual": pixel_perceptual_value,
            "latent_consistency": latent_value,
            "latent_rms": jnp.mean(
                jnp.sqrt(jnp.mean(jnp.square(sphere_latents), axis=(-2, -1)))
            ),
        }
        return total, metrics

    return loss_fn
