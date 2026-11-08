<h1 align='center'> joint </h1>
<h2 align='center'> Simple multi-modal generative models in JAX </h2>

Basic code to implement a multi-modal generative model. A DiT model is wrapped in encoder-decoder modules for each modality, which are trained end to end.

The code is designed to be modular for building models with UNet and Transformer architectures, which parameterise a deep generative model (e.g. a Sphere Encoder, Flow Matching or Flow Map model).

<p>
    <img src="assets/zuryon.png" alt>
    <em>A diagram of a multi-modal flow matching model that was built with this code, mapping multi-modal output from a cosmological N-body simulation.</em>
</p>

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
