# Research: JAX DiT Training Implementation

## Decisions & Rationale

### 1. Neural Network Framework: Flax NNX
- **Decision**: Use **Flax NNX** for model implementation.
- **Rationale**: NNX provides reference-based semantics which simplifies state management compared to the functional-only Linen API. Specifically, implementing Exponential Moving Average (EMA) using `nnx.EMA` is cleaner and integrates directly with the model instance.
- **Alternatives Considered**: Flax Linen (rejected due to more complex EMA/state handling), Equinox (rejected to stay within the preferred Flax ecosystem for this project).

### 2. Latent Encoding Strategy: Hybrid with JIT-Integrated VAE
- **Decision**: Implement on-the-fly encoding as part of the JIT-compiled training step (GPU-side) using `diffusers.FlaxAutoencoderKL`.
- **Rationale**: Calling the VAE encoder inside Grain workers (CPU) would create a significant PCIe bottleneck. Modern GPUs handle VAE encoding much faster than CPUs, and keeping it in the JIT graph ensures end-to-end optimization.
- **Performance Note**: For large-scale training beyond CIFAR, we will provide a utility script to pre-compute latents and save them as `ArrayRecord` for Grain to load directly.

### 3. Numerical Precision: BF16 Mixed Precision
- **Decision**: Use `bfloat16` for activations and gradients, while keeping master weights in `float32`.
- **Rationale**: BF16 provides the same dynamic range as FP32, avoiding the need for complex loss scaling (required by FP16) while providing 2x throughput on NVIDIA Ampere+ and TPUs.
- **Constraint**: Softmax and loss calculations MUST be performed in FP32 to avoid numerical instability.

### 4. Evaluation Metrics: `jax-fid`
- **Decision**: Integrate `jax-fid` for Fréchet Inception Distance calculation.
- **Rationale**: `jax-fid` is a numerically accurate port of the standard `pytorch-fid`, ensuring our research results are comparable with existing literature.
- **Implementation**: We will pre-calculate CIFAR-10 statistics and compare them against periodically generated samples during the validation phase.

### 5. Data Loading: Grain
- **Decision**: Use **Grain** for the primary data pipeline.
- **Rationale**: Grain offers better performance and more robust determinism than `tf.data` or standard JAX data loaders. It handles shuffling and batching efficiently across multiple CPU cores.
- **Integration**: We will use `MapTransform` for basic image preprocessing (normalization to `[-1, 1]`) and `BatchTransform` for efficient batch delivery.

## Dependencies Verification
- **Confirmed**: `diffusers` has a robust `FlaxAutoencoderKL` implementation.
- **Confirmed**: `nnx.EMA` is the recommended path for weight averaging in the newest Flax versions.
- **Confirmed**: `jax-fid` supports TPU/GPU acceleration.
