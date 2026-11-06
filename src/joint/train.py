import inspect
from pathlib import Path
from typing import Union, Optional, Iterator
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxtyping import PRNGKeyArray, Scalar, Array, Float, Int, Bool

from .generative.flow.train import _loss_vs_t_bins, get_time
from .generative.flow_matching import SolutionKwargs, OptimalTransportPath
from .utils import apply_ema, tree_random_normal_like, make_layernorm_decay_mask


def save_training_state(
    checkpoint_path: Union[str, Path],
    model: eqx.Module,
    ema_model: Optional[eqx.Module],
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    train_step: int,
) -> None:
    """Serialize model/optimizer training state to disk.

    Inputs:
        checkpoint_path (str | Path): Output checkpoint path.
        model (eqx.Module): Current trainable model.
        ema_model (eqx.Module): EMA model snapshot (or ``None`` when disabled).
        optimiser (optax.GradientTransformation): Optimizer transform object.
        optimiser_state (optax.OptState): Current optimizer internal state.
        train_step (int): Current global optimization step.

    Outputs:
        None: Writes a serialized checkpoint file.
    """
    state = {
        "model": model,
        "ema_model": ema_model,
        "optimiser": optimiser,
        "optimiser_state": optimiser_state,
        "train_step": jnp.asarray(train_step, dtype=jnp.int32),
    }
    eqx.tree_serialise_leaves(checkpoint_path, state)


def load_training_state(
    checkpoint_path: Union[str, Path],
    model: eqx.Module,
    optimiser: optax.GradientTransformation,
    optimiser_state: optax.OptState,
    train_step: int = 0,
    *,
    use_ema: bool = True,
) -> tuple[
    eqx.Module, 
    Optional[eqx.Module],
    optax.GradientTransformation, 
    optax.OptState, 
    int
]:
    """Load model/optimizer state from a serialized checkpoint.

    Inputs:
        checkpoint_path (str | Path): Input checkpoint file.
        model (eqx.Module): Template model structure.
        optimiser (optax.GradientTransformation): Template optimizer object.
        optimiser_state (optax.OptState): Template optimizer state.
        train_step (int): Fallback step value when building template state.
        use_ema (bool): Whether EMA was enabled when the checkpoint was saved.

    Outputs:
        tuple: ``(model, ema_model, optimiser, optimiser_state, train_step)``.
    """
    template = {
        "model": model,
        "ema_model": model if use_ema else None,
        "optimiser": optimiser,
        "optimiser_state": optimiser_state,
        "train_step": jnp.asarray(train_step, dtype=jnp.int32),
    }
    state = eqx.tree_deserialise_leaves(checkpoint_path, template)
    return (
        state["model"],
        state["ema_model"],
        state["optimiser"],
        state["optimiser_state"],
        int(state["train_step"]),
    )


def get_optimiser(
    model: eqx.Module,
    train_config,
) -> optax.GradientTransformation:
    """Build the optimiser transform configured for training.

    Inputs:
        model (eqx.Module): Model used to derive optional decay masks.
        train_config: Training config node with optimiser/schedule settings.

    Outputs:
        optax.GradientTransformation: Optimiser, optionally wrapped with
            warmup/cosine scheduling and global-norm clipping.
    """

    def _factory_accepts_kwarg(factory, kwarg: str) -> bool:
        """Return whether ``factory`` can be called with the given keyword."""
        try:
            inspect.signature(factory).bind_partial(**{kwarg: None})
        except (TypeError, ValueError):
            return False
        return True

    opt_kwargs = dict(train_config.opt_kwargs)

    if "mask" not in opt_kwargs and _factory_accepts_kwarg(
        train_config.opt_factory, "mask"
    ):
        opt_kwargs["mask"] = make_layernorm_decay_mask(model)

    use_lr_schedule = bool(getattr(train_config, "use_lr_schedule", True))

    learning_rate = opt_kwargs.get(
        "learning_rate", float(getattr(train_config, "learning_rate", 5e-4))
    )

    if use_lr_schedule:
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "Warmup/cosine schedule requires numeric learning_rate in train_config.opt_kwargs."
            )
        warmup_steps = int(getattr(train_config, "warmup_steps", 1_000))
        warmup_steps = min(max(0, warmup_steps), max(0, int(train_config.n_steps) - 1))
        # Optax's total schedule length already includes the warmup.
        decay_steps = max(1, int(train_config.n_steps))
        lr_end_value = float(
            getattr(train_config, "lr_end_value", float(learning_rate) * 0.1)
        )
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(learning_rate),
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=lr_end_value,
        )

    opt_kwargs["learning_rate"] = learning_rate
    base_opt = train_config.opt_factory(**opt_kwargs)
    clip_grad_norm = float(getattr(train_config, "clip_grad_norm", 1.0))
    if clip_grad_norm > 0.0:
        return optax.chain(optax.clip_by_global_norm(clip_grad_norm), base_opt)
    return base_opt


def _mse(y_, y):
    return jnp.mean(jnp.square(jnp.subtract(y_, y)))


def _sample_conditioning_mask(
    key: PRNGKeyArray,
    n_modalities: int,
    *,
    conditional_training_probability: float,
) -> Optional[Bool[Array, "m"]]:
    """Sample a batch-level subset of modalities to use as fixed conditions.

    Inputs:
        key (PRNGKeyArray): Random key.
        n_modalities (int): Number of modalities in the batch structure.
        conditional_training_probability (float): Probability of using a masked
            conditional batch instead of an unconditional one.

    Outputs:
        Array | None: Boolean modality mask where ``True`` marks conditioning
        modalities, or ``None`` when conditional training is disabled.
    """
    if conditional_training_probability <= 0.0 or n_modalities < 2:
        return None

    key_use, key_perm, key_count = jr.split(key, 3)
    use_conditional = jr.bernoulli(key_use, conditional_training_probability)
    n_conditioning = jr.randint(key_count, (), minval=1, maxval=n_modalities)
    permutation = jr.permutation(key_perm, n_modalities)
    ranks = jnp.argsort(permutation)
    conditioning_mask = ranks < n_conditioning
    return jnp.where(use_conditional, conditioning_mask, jnp.zeros_like(conditioning_mask))


def loss_fn(
    model: eqx.Module,
    path: OptimalTransportPath,
    d: dict[str, Float[Array, "..."]],
    z: dict[str, Float[Array, "..."]],
    t: Scalar,
    keys: PRNGKeyArray,
    *,
    t0: float,
    t1: float,
    n_t_bins: int,
    conditioning_mask: Optional[Bool[Array, "m"]] = None,
) -> tuple[Scalar, dict[str, Scalar], Float[Array, "b"], Float[Array, "b"]]:
    """Compute flow-matching objective across a batch of modalities.

    Inputs:
        model (eqx.Module): Velocity model.
        path (OptimalTransportPath): Flow path with ``psi_t`` and ``d_psi_t_dt``.
        d (dict[str, Array]): Clean batch modalities.
        z (dict[str, Array]): Noise batch modalities with same structure as ``d``.
        t (Array): Per-example diffusion/flow times.
        keys (PRNGKeyArray): Per-example model random keys.
        conditioning_mask (Array | None): Optional batch-level modality mask where
            ``True`` marks conditioning modalities kept clean at every time.

    Outputs:
        tuple: ``(loss, per_modality_loss, t_bin_loss, t_bin_count)``.
    """
    # Interpolate generated modalities; conditioning inputs stay clean.
    modality_names = tuple(sorted(d.keys()))
    attention_mask = None
    generated_weights = None

    if conditioning_mask is not None:
        attention_mask = model.build_directional_attention_mask(
            conditioning_mask,
            keys=modality_names,
        )
        generated_weights = jnp.logical_not(conditioning_mask)

    def _single_loss_fn(t, d, z, key):
        u = jax.tree.map(lambda x, z: path.d_psi_t_dt(t, x, z), d, z)
        psi = jax.tree.map(lambda x, z: path.psi_t(t, x, z), d, z)
        if conditioning_mask is not None:
            psi = {
                name: jnp.where(conditioning_mask[i], d[name], psi[name])
                for i, name in enumerate(modality_names)
            }

        u_ = model(psi, t, key=key, attention_mask=attention_mask)

        per_modality = jax.tree.map(_mse, u, u_)
        modality_losses = jnp.stack([per_modality[name] for name in modality_names])

        if generated_weights is None:
            l = jnp.sum(modality_losses)
        else:
            loss_weights = generated_weights.astype(modality_losses.dtype)
            n_generated = jnp.maximum(jnp.sum(loss_weights), 1.0)
            l = (
                jnp.sum(modality_losses * loss_weights)
                * (loss_weights.shape[0] / n_generated)
            )

        return l, per_modality

    per_sample_loss, per_sample_modality_loss = jax.vmap(_single_loss_fn)(t, d, z, keys)

    loss = jnp.mean(per_sample_loss)
    per_modality_loss = jax.tree.map(jnp.mean, per_sample_modality_loss)

    t_bin_loss, t_bin_count = _loss_vs_t_bins(
        per_sample_loss, t, t0=t0, t1=t1, n_bins=n_t_bins
    )

    return loss, per_modality_loss, t_bin_loss, t_bin_count


@eqx.filter_jit(donate="all")
def make_step(
    v: eqx.Module, 
    path: OptimalTransportPath, 
    key: PRNGKeyArray, 
    d: dict[str, Float[Array, "n ..."] | Int[Array, "n ..."]],
    opt_state: optax.OptState, 
    opt: optax.GradientTransformation, 
    v_ema: Optional[eqx.Module] = None, 
    *, 
    solution_kwargs: SolutionKwargs,
    time_sampling_distribution: str = "uniform",
    time_sampling_beta_alpha: float = 1.0,
    time_sampling_beta_beta: float = 1.0,
    time_sampling_stratified: bool = True,
    n_t_bins: int = 4,
    conditional_training_probability: float = 0.0,
) -> tuple[
    eqx.Module,
    Scalar,
    optax.OptState,
    Optional[eqx.Module],
    dict[str, Float[Array, "..."] | dict[str, Float[Array, ""]]],
]:
    """Run one optimization step for the flow model.

    Inputs:
        v (eqx.Module): Current model parameters.
        path (OptimalTransportPath): Flow path used for training targets.
        key (PRNGKeyArray): Step-level random key.
        d (dict[str, Array]): Batch modalities.
        opt_state (optax.OptState): Current optimizer state.
        opt (optax.GradientTransformation): Optimizer transform.
        v_ema (eqx.Module | None): Optional EMA model.
        solution_kwargs (SolutionKwargs): Time interval configuration.
        time_sampling_distribution (str): ``uniform`` or ``beta`` time sampling.
        time_sampling_beta_alpha (float): Beta alpha parameter when using Beta sampling.
        time_sampling_beta_beta (float): Beta beta parameter when using Beta sampling.
        time_sampling_stratified (bool): Whether to stratify uniform time sampling.
        n_t_bins (int): Number of bins used for logging loss versus time.
        conditional_training_probability (float): Probability of sampling a
            directional conditional-training batch.

    Outputs:
        tuple: ``(updated_model, loss, updated_opt_state, updated_ema_model, metrics)``.
    """
    n_batch = len(next(iter(d.values())))
    key_t, key_z, key_step, key_conditioning = jr.split(key, 4)

    keys = jr.split(key_step, n_batch)

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

    z = tree_random_normal_like(d, key_z)

    conditioning_mask = _sample_conditioning_mask(
        key_conditioning,
        n_modalities=len(d),
        conditional_training_probability=conditional_training_probability,
    )

    def _loss_with_aux(model, path, d, z, t, keys, conditioning_mask):
        loss, per_modality_loss, t_bin_loss, t_bin_count = loss_fn(
            model,
            path,
            d,
            z,
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

    grad_fn = eqx.filter_value_and_grad(_loss_with_aux, has_aux=True)
    (loss, metrics), grads = grad_fn(v, path, d, z, t, keys, conditioning_mask)

    params = eqx.filter(v, eqx.is_inexact_array)
    updates, opt_state = opt.update(grads, opt_state, params)
    v = eqx.apply_updates(v, updates)

    if v_ema is not None:
        v_ema = apply_ema(v_ema, v)

    return v, loss, opt_state, v_ema, metrics


@eqx.filter_jit
def evaluate(
    v: eqx.Module,
    path: OptimalTransportPath,
    key: PRNGKeyArray,
    d: dict[str, Float[Array, "n ..."] | Int[Array, "n ..."]],
    *,
    solution_kwargs: SolutionKwargs,
    time_sampling_distribution: str = "uniform",
    time_sampling_beta_alpha: float = 1.0,
    time_sampling_beta_beta: float = 1.0,
    time_sampling_stratified: bool = True,
    conditional_training_probability: float = 0.0,
) -> Scalar:
    """Evaluate batch loss in inference mode without parameter updates.

    Inputs:
        v (eqx.Module): Model to evaluate.
        path (OptimalTransportPath): Flow path used for target construction.
        key (PRNGKeyArray): Evaluation random key.
        d (dict[str, Array]): Evaluation batch.
        solution_kwargs (SolutionKwargs): Time interval configuration.
        time_sampling_distribution (str): ``uniform`` or ``beta`` time sampling.
        time_sampling_beta_alpha (float): Beta alpha parameter when using Beta sampling.
        time_sampling_beta_beta (float): Beta beta parameter when using Beta sampling.
        time_sampling_stratified (bool): Whether to stratify uniform time sampling.
        conditional_training_probability (float): Probability of sampling a
            directional conditional-training batch.

    Outputs:
        Scalar: Batch loss value.
    """
    n_batch = len(next(iter(d.values())))
    key_t, key_z, key_step, key_conditioning = jr.split(key, 4)

    v = eqx.nn.inference_mode(v)

    keys = jr.split(key_step, n_batch)
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
    z = tree_random_normal_like(d, key_z)
    conditioning_mask = _sample_conditioning_mask(
        key_conditioning,
        len(d),
        conditional_training_probability=conditional_training_probability,
    )

    loss, _, _, _ = loss_fn(
        v,
        path,
        d,
        z,
        t,
        keys,
        t0=solution_kwargs.t0,
        t1=solution_kwargs.t1,
        n_t_bins=1,
        conditioning_mask=conditioning_mask,
    )

    return loss


def evaluate_validation_set(
    model: eqx.Module,
    path: OptimalTransportPath,
    val_loader: Iterator,
    n_val_batches: int,
    *,
    key: PRNGKeyArray,
    solution_kwargs: SolutionKwargs,
    time_sampling_distribution: str = "uniform",
    time_sampling_beta_alpha: float = 1.0,
    time_sampling_beta_beta: float = 1.0,
    time_sampling_stratified: bool = True,
    conditional_training_probability: float = 0.0,
) -> Scalar:
    """Average evaluation loss over multiple validation batches.

    Inputs:
        model (eqx.Module): Model to evaluate.
        path (OptimalTransportPath): Flow path for target construction.
        val_loader (Iterator[dict[str, Array]]): Validation batch iterator.
        n_val_batches (int): Number of batches to evaluate.
        key (PRNGKeyArray): Random key for evaluation.
        solution_kwargs (SolutionKwargs): Time interval configuration.
        time_sampling_distribution (str): ``uniform`` or ``beta`` time sampling.
        time_sampling_beta_alpha (float): Beta alpha parameter when using Beta sampling.
        time_sampling_beta_beta (float): Beta beta parameter when using Beta sampling.
        time_sampling_stratified (bool): Whether to stratify uniform time sampling.
        conditional_training_probability (float): Probability of sampling a
            directional conditional-training batch.

    Outputs:
        Scalar: Mean validation loss across evaluated batches.
    """
    losses = []
    for _ in range(n_val_batches):
        key, key_eval = jr.split(key)
        losses.append(
            evaluate(
                model,
                path,
                key_eval,
                next(val_loader),
                solution_kwargs=solution_kwargs,
                time_sampling_distribution=time_sampling_distribution,
                time_sampling_beta_alpha=time_sampling_beta_alpha,
                time_sampling_beta_beta=time_sampling_beta_beta,
                time_sampling_stratified=time_sampling_stratified,
                conditional_training_probability=conditional_training_probability,
            )
        )
    return jnp.mean(jnp.stack(losses))


def build_make_step(loss_fn):
    """Build a generic optimization step from a family-specific loss function."""

    @eqx.filter_jit(donate="all")
    def make_step(
        model: eqx.Module,
        key: PRNGKeyArray,
        batch: dict[str, Float[Array, "n ..."] | Int[Array, "n ..."]],
        opt_state: optax.OptState,
        opt: optax.GradientTransformation,
        ema_model: Optional[eqx.Module] = None,
    ) -> tuple[
        eqx.Module,
        Scalar,
        optax.OptState,
        Optional[eqx.Module],
        dict[str, Float[Array, "..."] | dict[str, Float[Array, ""]]],
    ]:
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(model, batch, key)

        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = opt.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)

        if ema_model is not None:
            ema_model = apply_ema(ema_model, model)

        return model, loss, opt_state, ema_model, metrics

    return make_step


def build_evaluate(loss_fn):
    """Build a generic evaluation function from a family-specific loss function."""

    @eqx.filter_jit
    def evaluate(
        model: eqx.Module,
        key: PRNGKeyArray,
        batch: dict[str, Float[Array, "n ..."] | Int[Array, "n ..."]],
    ) -> Scalar:
        model = eqx.nn.inference_mode(model)
        loss, _ = loss_fn(model, batch, key)
        return loss

    return evaluate


def evaluate_validation_batches(
    evaluate_fn,
    model: eqx.Module,
    val_loader: Iterator,
    n_val_batches: int,
    *,
    key: PRNGKeyArray,
) -> Scalar:
    """Average generic evaluation loss over several validation batches."""
    losses = []
    for _ in range(n_val_batches):
        key, key_eval = jr.split(key)
        losses.append(evaluate_fn(model, key_eval, next(val_loader)))
    return jnp.mean(jnp.stack(losses))
