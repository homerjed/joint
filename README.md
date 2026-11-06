<h1 align='center'> joint </h1>
<h2 align='center'> Simple multi-modal generative models in JAX </h2>

Basic code to implement a multi-modal generative model. A DiT model is wrapped in encoder-decoder modules for each modality, which are trained end to end.

The code is designed to be modular for building models with UNet and Transformer architectures, which parameterise a deep generative model (e.g. a Sphere Encoder, Flow Matching or Flow Map model).

<p>
    <img src="assets/zuryon.png" alt>
    <em>A diagram of a multi-modal flow matching model that was built with this code, mapping multi-modal output from a cosmological N-body simulation.</em>
</p>

## Flow maps and few-step sampling

The `flow_map` family implements the two-time parameterization
`X(s, t, x) = x + (t - s) v(s, t, x)` from
[Boffi, Albergo and Vanden-Eijnden (NeurIPS 2025)](https://arxiv.org/abs/2505.18825).
The diagonal `v(t, t, x)` learns the usual flow-matching velocity. An additional
self-distillation loss trains finite jumps, without a pretrained teacher:

| `train_config.flow_map_loss` | Off-diagonal objective |
| --- | --- |
| `lsd` (default) | Match the destination-time derivative of the map to its diagonal velocity. |
| `esd` | Match the Eulerian equation using start-time and spatial JVPs. |
| `psd` | Match one jump to the composition of two smaller jumps. |

The implementation follows the paper's stopped teacher gradients. PSD uses the
velocity-preconditioned residual, without division by the interval length, and
supports `flow_map_psd_split="uniform"` or `"midpoint"`. LSD and ESD use exact JAX
JVPs. The diagonal and distillation terms use the full batch with separate times
and fixed weights (`flow_map_diagonal_weight=1.0` and
`flow_map_distillation_weight=1.0`), rather than the paper's learned loss weights
and split batch. Setting the distillation weight to zero allows velocity warmup.

Train with either example script (use a new run name for a new model family):

```bash
uv run python -m scripts.main_1d --run_name flow_map_1d --model_family flow_map --flow_map_loss lsd --flow_map_steps 4
uv run python -m scripts.main --run_name flow_map_2d_psd --model_family flow_map --flow_map_loss psd --flow_map_psd_split midpoint --flow_map_steps 2
```

These use the existing training/data defaults; the options are persisted in
`out/<run_name>/config.json`. `--reload` resumes a run. Flow-map checkpoints have
an additional time embedder and cannot be loaded from flow-matching checkpoints.

`sampling_config.flow_map_steps` controls the family sampler (default: 4).
Sampling composes learned maps, with one network evaluation per step and no ODE
solver. The step count can also be changed after training using the direct API:

```python
from joint.generative.flow_map import single_sample_fn, single_sample_modalities_fn

# model is a trained flow_map family model; noise contains one unbatched sample.
one_step = single_sample_fn(model, noise, n_steps=1)
four_steps = single_sample_fn(model, noise, n_steps=4)
custom_grid = single_sample_fn(model, noise, times=[0.0, 0.1, 0.5, 1.0])

# Start x with noise and provide clean observations for a and b.
conditional = single_sample_modalities_fn(
    model, {"x": noise["x"], "a": observed_a, "b": observed_b},
    ["x"], n_steps=2,
)
```

Use `jax.vmap` for batches. `times` is the full integration grid, including both
endpoints; the sampler returns the final state. Conditional flow-map training
and sampling keep observed modalities clean and fixed at every stage. Only
generated modalities contribute to the losses. Flow matching uses the same
clean-conditioning convention during training and ODE sampling. Existing
flow-matching checkpoints trained with noisy conditioning need retraining or
fine-tuning for this convention; loading weights alone does not adapt them. Set
`train_config.conditional_training_probability` between 0 and 1 to mix
unconditional and conditional batches; the existing default of 1 trains only
conditional batches when multiple modalities are present. Sample quality still
needs to be measured on trained checkpoints; the tests verify mathematical
identities, gradients and integration, not benchmark performance.

## Inference notebooks

Use `scripts/inference.ipynb` for 2D runs and `scripts/inference_1d.ipynb` for 1D
runs. They load checkpoints, generate joint and conditional samples, and compare
parameter posteriors and predictive fields. Both support `flow_matching` and
`flow_map`, and resolve checkpoint paths from either the repository root or
`scripts/`. Set `run_name` to the run to inspect. For flow maps, set
`config.sampling_config.flow_map_steps` before building the family to change the
sampling cost.

The root `inference.ipynb` is a legacy duplicate of the 2D notebook, with no
unique analysis. It still has stale imports, an undefined checkpoint path and
a syntax error, and is retained only as a historical artifact. It is not needed
by the package or training scripts.

Run the focused flow-map checks with:

```bash
uv run python -m unittest discover -s tests -v
```
