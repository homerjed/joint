from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar

from ...utils import tree_random_normal_like
from .model import FlowMatchingModel
from .path import OptimalTransportPath
from .solvers import SolutionKwargs


def get_time(
    key: PRNGKeyArray,
    n: int,
    t0: float,
    t1: float,
    *,
    distribution: str = "uniform",
    beta_alpha: float = 1.0,
    beta_beta: float = 1.0,
    stratified: bool = True,
) -> Float[Array, "n"]:
    if distribution == "uniform":
        if stratified:
            u = jr.uniform(key, (n,), minval=0.0, maxval=1.0 / n)
            u = u + jnp.arange(n) / n
        else:
            u = jr.uniform(key, (n,), minval=0.0, maxval=1.0)
    elif distribution == "beta":
        u = jr.beta(key, beta_alpha, beta_beta, shape=(n,))
    else:
        raise ValueError(f"Unknown time sampling distribution: {distribution!r}")

    return t0 + (t1 - t0) * u


def _mse(y_pred, y_true):
    return jnp.mean(jnp.square(jnp.subtract(y_pred, y_true)))


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


def _loss_vs_t_bins(
    losses: Float[Array, "n"],
    t: Float[Array, "n"],
    *,
    t0: float,
    t1: float,
    n_bins: int,
) -> tuple[Float[Array, "b"], Float[Array, "b"]]:
    edges = jnp.linspace(t0, t1, n_bins + 1, dtype=t.dtype)
    means = []
    counts = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i < n_bins - 1:
            mask = (t >= lo) & (t < hi)
        else:
            mask = (t >= lo) & (t <= hi)
        count = jnp.sum(mask)
        weight = mask.astype(losses.dtype)
        total = jnp.sum(losses * weight)
        mean = jnp.where(count > 0, total / count, jnp.nan)
        means.append(mean)
        counts.append(count.astype(losses.dtype))
    return jnp.stack(means), jnp.stack(counts)


def compute_flow_matching_loss(
    model: FlowMatchingModel,
    path: OptimalTransportPath,
    batch: dict[str, Float[Array, "..."]],
    noise: dict[str, Float[Array, "..."]],
    t: Float[Array, "n"],
    keys: PRNGKeyArray,
    *,
    t0: float,
    t1: float,
    n_t_bins: int,
    conditioning_mask: Optional[Bool[Array, "m"]] = None,
) -> tuple[Scalar, dict[str, Scalar], Float[Array, "b"], Float[Array, "b"]]:
    """Match generated velocities while keeping conditioning inputs clean."""
    modality_names = tuple(sorted(batch.keys()))
    attention_mask = None
    generated_weights = None

    if conditioning_mask is not None:
        attention_mask = model.build_directional_attention_mask(
            conditioning_mask,
            keys=modality_names,
        )
        generated_weights = jnp.logical_not(conditioning_mask)

    def _single_loss_fn(t_value, clean_modalities, noise_modalities, model_key):
        targets = jax.tree.map(
            lambda x, z: path.d_psi_t_dt(t_value, x, z),
            clean_modalities,
            noise_modalities,
        )
        path_state = jax.tree.map(
            lambda x, z: path.psi_t(t_value, x, z),
            clean_modalities,
            noise_modalities,
        )
        if conditioning_mask is not None:
            path_state = {
                name: jnp.where(conditioning_mask[i], clean_modalities[name], path_state[name])
                for i, name in enumerate(modality_names)
            }

        predictions = model(path_state, t_value, key=model_key, attention_mask=attention_mask)
        per_modality = jax.tree.map(_mse, predictions, targets)
        modality_losses = jnp.stack([per_modality[name] for name in modality_names])

        if generated_weights is None:
            total = jnp.sum(modality_losses)
        else:
            loss_weights = generated_weights.astype(modality_losses.dtype)
            n_generated = jnp.maximum(jnp.sum(loss_weights), 1.0)
            total = (
                jnp.sum(modality_losses * loss_weights)
                * (loss_weights.shape[0] / n_generated)
            )

        return total, per_modality

    per_sample_loss, per_sample_modality_loss = jax.vmap(_single_loss_fn)(
        t,
        batch,
        noise,
        keys,
    )

    loss = jnp.mean(per_sample_loss)
    per_modality_loss = jax.tree.map(jnp.mean, per_sample_modality_loss)
    t_bin_loss, t_bin_count = _loss_vs_t_bins(
        per_sample_loss,
        t,
        t0=t0,
        t1=t1,
        n_bins=n_t_bins,
    )
    return loss, per_modality_loss, t_bin_loss, t_bin_count


def build_loss_fn(
    path: OptimalTransportPath,
    *,
    solution_kwargs: SolutionKwargs,
    train_config,
):
    time_sampling_distribution = str(
        getattr(train_config, "time_sampling_distribution", "uniform")
    )
    time_sampling_beta_alpha = float(
        getattr(train_config, "time_sampling_beta_alpha", 1.0)
    )
    time_sampling_beta_beta = float(
        getattr(train_config, "time_sampling_beta_beta", 1.0)
    )
    time_sampling_stratified = bool(
        getattr(train_config, "time_sampling_stratified", True)
    )
    n_t_bins = max(1, int(getattr(train_config, "n_t_bins", 4)))
    conditional_training_probability = float(
        getattr(train_config, "conditional_training_probability", 0.0)
    )

    def loss_fn(
        model: FlowMatchingModel,
        batch: dict[str, Float[Array, "n ..."]],
        key: PRNGKeyArray,
    ):
        n_batch = len(next(iter(batch.values())))
        key_t, key_noise, key_model, key_conditioning = jr.split(key, 4)
        keys = jr.split(key_model, n_batch)
        t = get_time(
            key_t,
            n_batch,
            t0=solution_kwargs.t0,
            t1=solution_kwargs.t1,
            distribution=time_sampling_distribution,
            beta_alpha=time_sampling_beta_alpha,
            beta_beta=time_sampling_beta_beta,
            stratified=time_sampling_stratified,
        )
        noise = tree_random_normal_like(batch, key_noise)
        conditioning_mask = _sample_conditioning_mask(
            key_conditioning,
            n_modalities=len(batch),
            conditional_training_probability=conditional_training_probability,
        )
        loss, per_modality_loss, t_bin_loss, t_bin_count = compute_flow_matching_loss(
            model,
            path,
            batch,
            noise,
            t,
            keys,
            t0=solution_kwargs.t0,
            t1=solution_kwargs.t1,
            n_t_bins=n_t_bins,
            conditioning_mask=conditioning_mask,
        )
        metrics = {
            "per_modality_loss": per_modality_loss,
            "t_bin_loss": t_bin_loss,
            "t_bin_count": t_bin_count,
        }
        return loss, metrics

    return loss_fn
