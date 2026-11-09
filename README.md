<h1 align='center'> joint </h1>
<h2 align='center'> Simple multi-modal generative models in JAX </h2>

Basic code to implement a multi-modal generative model. A DiT model is wrapped in encoder-decoder modules for each modality, which are trained end to end.

The code is designed to be modular for building models with UNet and Transformer architectures, which parameterise a deep generative model (e.g. a Sphere Encoder, Flow Matching or Flow Map model).

<p>
    <img src="assets/zuryon.png" alt>
    <em>A diagram of a multi-modal flow matching model that was built with this code, mapping multi-modal output from a cosmological N-body simulation.</em>
</p>

<!-- ## Examples

The default `uv sync` installs joint without the example dependencies. To install
the GalSBI examples, use Python 3.13 and run from the repository root:

```bash
uv sync --group examples
uv run --group examples python scripts/galsbi_dataset.py --n-samples 100 --n-pix 64
```

The [GalSBI script](scripts/galsbi_dataset.py) generates small sky patches with
paired `grizy` images, galaxy catalogues and population parameters, then saves a
NumPy dataset with catalogue padding masks and simulation metadata. Outputs go
to `out/galsbi/` relative to the working directory. Use `--n-samples` and
`--n-pix` to choose the dataset size and image resolution.

The [flow-map training script](scripts/galsbi_flow_map.py) loads that dataset
and jointly generates images, galaxy catalogues and population parameters. It
uses a shared per-galaxy MLP, masked multimodal attention, a fitted galaxy-count
distribution, and selectable PSD/ESD/LSD losses with one- or few-step sampling.

```bash
uv run --group examples python scripts/galsbi_flow_map.py --steps 1000 --loss psd
```

Training selects the latest archive under `out/galsbi/`; pass `--data path/to/data.npz`
to use a specific archive. Checkpoints, generated arrays and diagnostic plots are
saved under `out/galsbi_flow_map/`. Both scripts accept `--output-dir` and `--help`.

The default example uses GalSBI's image renderer. Pass `--source-extraction`
to the dataset script to also produce measured catalogues; this requires a separate
[SExtractor installation](https://sextractor.readthedocs.io/en/latest/Installing.html).
GalSBI downloads supporting data and may compile PyCosmo on its first run; see the
[GalSBI setup notes](https://cosmo-docs.phys.ethz.ch/galsbi/faq.html). -->

## Citations

```bibtex
@inproceedings{zhou2025transfusion,
    title     = {Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model},
    author    = {Chunting Zhou and Lili Yu and Arun Babu and Kushal Tirumala and Michihiro Yasunaga and Leonid Shamis and Jacob Kahn and Xuezhe Ma and Luke Zettlemoyer and Omer Levy},
    booktitle = {International Conference on Learning Representations},
    year      = {2025},
    url       = {https://arxiv.org/abs/2408.11039}
}
```

```bibtex
@inproceedings{boffi2025flowmaps,
    title     = {How to build a consistency model: Learning flow maps via self-distillation},
    author    = {Nicholas M. Boffi and Michael S. Albergo and Eric Vanden-Eijnden},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2025},
    url       = {https://arxiv.org/abs/2505.18825}
}
```

```bibtex
@inproceedings{lipman2023flowmatching,
    title     = {Flow Matching for Generative Modeling},
    author    = {Yaron Lipman and Ricky T. Q. Chen and Heli Ben-Hamu and Maximilian Nickel and Matt Le},
    booktitle = {International Conference on Learning Representations},
    year      = {2023},
    url       = {https://arxiv.org/abs/2210.02747}
}
```

```bibtex
@misc{yue2026sphereencoder,
    title         = {Image Generation with a Sphere Encoder},
    author        = {Kaiyu Yue and Menglin Jia and Ji Hou and Tom Goldstein},
    year          = {2026},
    eprint        = {2602.15030},
    archivePrefix = {arXiv},
    url           = {https://arxiv.org/abs/2602.15030}
}
```
