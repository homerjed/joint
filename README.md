<h1 align='center'> joint </h1>
<h2 align='center'> Simple multi-modal generative models in JAX </h2>

Basic code to implement a multi-modal generative model. A DiT model is wrapped in encoder-decoder modules for each modality, which are trained end to end.

The code is designed to be modular for building models with UNet and Transformer architectures, which parameterise a deep generative model (e.g. a Sphere Encoder, Flow Matching or Flow Map model).

<p>
    <img src="assets/zuryon.png" alt>
    <em>A diagram of a multi-modal flow matching model that was built with this code, mapping multi-modal output from a cosmological N-body simulation.</em>
</p>
