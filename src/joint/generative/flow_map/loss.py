"""Self-distillation objectives from Boffi et al. (NeurIPS 2025).

See https://arxiv.org/abs/2505.18825, Eqs. (12)--(15), (22), and (94).
The diagonal learns a velocity by flow matching; LSD, ESD, or PSD trains its
two-time flow map. Teachers use the current parameters with stopped gradients.
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr

from ...utils import tree_random_normal_like
from ..flow_matching.loss import (
    _loss_vs_t_bins,
    _sample_conditioning_mask,
    get_time,
)


def _clamp_conditioning(state, reference, conditioning_mask):
    if conditioning_mask is None:
        return state
    return {
        name: jnp.where(conditioning_mask[i], reference[name], state[name])
        for i, name in enumerate(sorted(state))
    }


def _generated_only(values, conditioning_mask):
    if conditioning_mask is None:
        return values
    return {
        name: jnp.where(conditioning_mask[i], jnp.zeros_like(value), value)
        for i, (name, value) in enumerate(sorted(values.items()))
    }


def _raw_validity_masks(model, modalities, masks):
    if masks is None:
        return None
    if set(masks).difference(modalities):
        raise ValueError("Mask keys must name modalities in the state.")
    tokenizer = model.backbone.tokenizer
    return {
        name: (
            tokenizer.raw_validity_mask(name, masks[name])
            if name in masks else jnp.ones_like(value, dtype=jnp.bool_)
        )
        for name, value in modalities.items()
    }


def _zero_padding(values, validity):
    if validity is None:
        return values
    return jax.tree.map(lambda value, valid: jnp.where(valid, value, 0), values, validity)


def distillation_residual(
    model,
    x,
    s,
    t,
    *,
    loss_type="lsd",
    split_fraction=0.5,
    attention_mask=None,
    key=None,
    conditioning_mask=None,
    masks=None,
):
    """Return one sample's LSD, ESD, or velocity-preconditioned PSD residual.

    ``model(x, s, t)`` predicts the average velocity in
    ``X(s,t,x) = x + (t-s) * v(s,t,x)``. Partial time derivatives hold ``x``
    fixed, and teacher/spatial-JVP gradients follow Eq. (94) of the paper.
    PSD uses ``u = s + split_fraction * (t-s)`` and drops the ``(t-s)**2``
    loss scale, avoiding divisions by small time intervals.

    A conditioning mask follows sorted modality names. Conditioned leaves in
    ``x`` are clean observations, held fixed at intermediate and final times;
    their residuals are zero. Use the same directional attention mask as in
    the diagonal loss and conditional sampler. ``masks`` contains unbatched
    token-validity masks; padded raw entries and their tangents stay zero.
    """
    loss_type = str(loss_type).lower()
    if loss_type not in {"lsd", "esd", "psd"}:
        raise ValueError("flow_map_loss must be 'lsd', 'esd', or 'psd'.")
    dtype = jnp.result_type(s, t, jnp.float32)
    s, t = jnp.asarray(s, dtype=dtype), jnp.asarray(t, dtype=dtype)
    validity = _raw_validity_masks(model, x, masks)
    x = _zero_padding(x, validity)
    mask_kwargs = {} if masks is None else {"masks": masks}

    def velocity(state, start, end):
        state = _zero_padding(state, validity)
        prediction = model(
            state, start, end, attention_mask=attention_mask, key=key, **mask_kwargs
        )
        return _zero_padding(prediction, validity)

    def flow_map(state, start, end):
        result = jax.tree.map(
            lambda value, v: value + (end - start) * v,
            state,
            velocity(state, start, end),
        )
        result = _clamp_conditioning(result, state, conditioning_mask)
        return _zero_padding(result, validity)

    if loss_type == "lsd":
        endpoint, derivative = jax.jvp(
            lambda end: flow_map(x, s, end), (t,), (jnp.ones_like(t),)
        )
        target = jax.tree.map(jax.lax.stop_gradient, velocity(endpoint, t, t))
        residual = jax.tree.map(jnp.subtract, derivative, target)
    elif loss_type == "esd":
        drift = _generated_only(velocity(x, s, s), conditioning_mask)
        _, time_derivative = jax.jvp(
            lambda start: flow_map(x, start, t), (s,), (jnp.ones_like(s),)
        )
        _, spatial_derivative = jax.jvp(
            lambda state: flow_map(state, s, t), (x,), (drift,)
        )
        spatial_derivative = jax.tree.map(
            jax.lax.stop_gradient, spatial_derivative
        )
        residual = jax.tree.map(jnp.add, time_derivative, spatial_derivative)
    else:
        u = s + split_fraction * (t - s)
        first_velocity = velocity(x, s, u)
        midpoint = jax.tree.map(
            lambda value, v: value + (u - s) * v, x, first_velocity
        )
        midpoint = _clamp_conditioning(midpoint, x, conditioning_mask)
        midpoint = _zero_padding(midpoint, validity)
        second_velocity = velocity(midpoint, u, t)
        target = jax.tree.map(
            lambda first, second: jax.lax.stop_gradient(
                split_fraction * first + (1.0 - split_fraction) * second
            ),
            first_velocity,
            second_velocity,
        )
        residual = jax.tree.map(jnp.subtract, velocity(x, s, t), target)

    return _zero_padding(_generated_only(residual, conditioning_mask), validity)


def build_loss_fn(path, *, train_config, t0=0.0, t1=1.0):
    """Build diagonal flow matching plus an off-diagonal distillation loss.

    Each batch supplies both terms with independently sampled times. The
    off-diagonal pairs are uniform on ``t0 <= s <= t <= t1``. Diagonal times
    use the existing flow-matching time sampling settings. Fixed nonnegative
    loss weights are used; this does not reproduce the paper's optional
    learned loss weighting or its diagonal/off-diagonal batch partition.
    """
    loss_type = str(getattr(train_config, "flow_map_loss", "lsd")).lower()
    psd_split = str(getattr(train_config, "flow_map_psd_split", "uniform")).lower()
    diagonal_weight = float(getattr(train_config, "flow_map_diagonal_weight", 1.0))
    distillation_weight = float(
        getattr(train_config, "flow_map_distillation_weight", 1.0)
    )
    probability = float(
        getattr(train_config, "conditional_training_probability", 0.0)
    )
    distribution = str(getattr(train_config, "time_sampling_distribution", "uniform"))
    beta_alpha = float(getattr(train_config, "time_sampling_beta_alpha", 1.0))
    beta_beta = float(getattr(train_config, "time_sampling_beta_beta", 1.0))
    stratified = bool(getattr(train_config, "time_sampling_stratified", True))
    n_t_bins = int(getattr(train_config, "n_t_bins", 4))

    if loss_type not in {"lsd", "esd", "psd"}:
        raise ValueError("flow_map_loss must be 'lsd', 'esd', or 'psd'.")
    if psd_split not in {"uniform", "midpoint"}:
        raise ValueError("flow_map_psd_split must be 'uniform' or 'midpoint'.")
    if not math.isfinite(diagonal_weight) or diagonal_weight <= 0.0:
        raise ValueError("flow_map_diagonal_weight must be finite and positive.")
    if not math.isfinite(distillation_weight) or distillation_weight < 0.0:
        raise ValueError("flow_map_distillation_weight must be finite and nonnegative.")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("conditional_training_probability must lie in [0, 1].")
    if not 0.0 <= t0 < t1 <= 1.0:
        raise ValueError("Flow-map training requires 0 <= t0 < t1 <= 1.")
    if distribution not in {"uniform", "beta"}:
        raise ValueError("time_sampling_distribution must be 'uniform' or 'beta'.")
    if not all(math.isfinite(v) and v > 0.0 for v in (beta_alpha, beta_beta)):
        raise ValueError("Beta time-sampling parameters must be finite and positive.")
    if n_t_bins < 1:
        raise ValueError("n_t_bins must be positive.")

    def loss_fn(model, batch, key, *, masks=None, conditioning_mask=None):
        """Evaluate the loss with optional batched token masks.

        ``masks[name]`` has shape ``(batch, sequence_length)``; omitted
        modalities are fully valid. An explicit shared ``conditioning_mask``
        follows sorted modality names and overrides random conditioning.
        """
        if not batch:
            raise ValueError("A flow-map training batch must contain modalities.")
        names = tuple(sorted(batch))
        n_batch = len(batch[names[0]])
        if n_batch < 1 or any(len(value) != n_batch for value in batch.values()):
            raise ValueError("Modalities must have the same nonempty batch dimension.")
        if masks is not None:
            if set(masks).difference(names):
                raise ValueError("Mask keys must name modalities in the batch.")
            if any(value.ndim < 1 or value.shape[0] != n_batch for value in masks.values()):
                raise ValueError("Token masks must have the same batch dimension as the data.")
        key_t, key_pair, key_split, key_noise, key_model, key_mask = jr.split(key, 6)
        times = get_time(
            key_t, n_batch, t0, t1,
            distribution=distribution,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            stratified=stratified,
        )
        pairs = jnp.sort(
            jr.uniform(key_pair, (n_batch, 2), minval=t0, maxval=t1), axis=-1
        )
        splits = (
            jr.uniform(key_split, (n_batch,))
            if psd_split == "uniform"
            else jnp.full((n_batch,), 0.5)
        )
        noise = tree_random_normal_like(batch, key_noise)
        model_keys = jr.split(key_model, n_batch)
        if conditioning_mask is None:
            conditioning_mask = _sample_conditioning_mask(
                key_mask, len(names), conditional_training_probability=probability
            )
        else:
            conditioning_mask = jnp.asarray(conditioning_mask, dtype=jnp.bool_)
            if conditioning_mask.shape != (len(names),):
                raise ValueError("conditioning_mask must have one entry per sorted modality.")
        attention_mask = (
            model.build_directional_attention_mask(conditioning_mask, keys=names)
            if conditioning_mask is not None
            else None
        )

        def reduce_modalities(residual, validity):
            if validity is None:
                values = {
                    name: jnp.mean(jnp.square(value))
                    for name, value in residual.items()
                }
            else:
                values = {
                    name: jnp.sum(jnp.square(jnp.where(validity[name], value, 0)))
                    / jnp.maximum(jnp.sum(validity[name]), 1)
                    for name, value in residual.items()
                }
            total = sum(values.values())
            if conditioning_mask is not None:
                # Match the existing modality-balanced FM loss normalization.
                n_generated = jnp.maximum(jnp.sum(~conditioning_mask), 1)
                total = total * (len(names) / n_generated)
            return total, values

        def single_loss(clean, normal, time, pair, fraction, model_key, sample_masks):
            validity = _raw_validity_masks(model, clean, sample_masks)
            # Clear inputs before arithmetic: even NaNs in padded rows are inert.
            clean = _zero_padding(clean, validity)
            normal = _zero_padding(normal, validity)
            mask_kwargs = {} if sample_masks is None else {"masks": sample_masks}

            def path_state(at):
                state = jax.tree.map(
                    lambda value, z: path.psi_t(at, value, z), clean, normal
                )
                state = _clamp_conditioning(state, clean, conditioning_mask)
                return _zero_padding(state, validity)

            diagonal_key, off_diagonal_key = jr.split(model_key)
            prediction = model(
                path_state(time), time, time,
                attention_mask=attention_mask, key=diagonal_key, **mask_kwargs,
            )
            target = jax.tree.map(
                lambda value, z: path.d_psi_t_dt(time, value, z), clean, normal
            )
            diagonal_residual = _generated_only(
                jax.tree.map(jnp.subtract, prediction, target), conditioning_mask
            )
            diagonal_residual = _zero_padding(diagonal_residual, validity)
            diagonal, diagonal_modalities = reduce_modalities(diagonal_residual, validity)
            if distillation_weight > 0.0:
                residual = distillation_residual(
                    model, path_state(pair[0]), pair[0], pair[1],
                    loss_type=loss_type,
                    split_fraction=fraction,
                    attention_mask=attention_mask,
                    key=off_diagonal_key,
                    conditioning_mask=conditioning_mask,
                    masks=sample_masks,
                )
                distilled, distilled_modalities = reduce_modalities(residual, validity)
            else:
                distilled = jnp.zeros_like(diagonal)
                distilled_modalities = jax.tree.map(jnp.zeros_like, diagonal_modalities)
            per_modality = jax.tree.map(
                lambda fm, sd: diagonal_weight * fm + distillation_weight * sd,
                diagonal_modalities, distilled_modalities,
            )
            return diagonal, distilled, per_modality

        diagonal, distilled, per_modality = jax.vmap(
            single_loss, in_axes=(0, 0, 0, 0, 0, 0, None if masks is None else 0)
        )(
            batch, noise, times, pairs, splits, model_keys, masks
        )
        per_sample = diagonal_weight * diagonal + distillation_weight * distilled
        # Time-bin diagnostics concern the diagonal, whose time is unambiguous.
        t_bin_loss, t_bin_count = _loss_vs_t_bins(
            diagonal, times, t0=t0, t1=t1, n_bins=n_t_bins
        )
        return jnp.mean(per_sample), {
            "per_modality_loss": jax.tree.map(jnp.mean, per_modality),
            "t_bin_loss": t_bin_loss,
            "t_bin_count": t_bin_count,
            "flow_matching_loss": jnp.mean(diagonal),
            "distillation_loss": jnp.mean(distilled),
        }

    return loss_fn


__all__ = ["build_loss_fn", "distillation_residual"]
