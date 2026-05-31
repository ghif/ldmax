# CelebA TPU Optimization Analysis

## Current State

The current training pipeline for `configs/celeba_tpu_b384.yaml` utilizes JAX and Flax NNX with a batch size of 384. The multi-device setup relies on `jax.sharding.NamedSharding` for data parallelism. A `DevicePrefetcher` is employed to asynchronously transfer data batches from the host to the TPU devices.

## Optimization Opportunities for TPU Parallel Computation

### 1. Host-Device Transfer Bottlenecks (RNG State)
**Issue:** Inside `src/scripts/train.py`, the training loop generates a new PRNG key on the host CPU every step using `rng_manager.next()` and passes it to `train_step` and `encode_fn`. This introduces a small but constant host-to-device communication latency at every step.
**Recommendation:** Move PRNG key splitting inside the compiled `@nnx.jit(train_step)`. Maintain a sharded RNG key state directly on the TPU devices and return the updated key from the `train_step`. This ensures that the entire step loop runs continuously without blocking on host randomness generation.

### 2. VAE Encoding Bottleneck
**Issue:** While `DevicePrefetcher` hides the latency of moving raw images to the device, the VAE encoding step (`encode_fn`) is performed synchronously inside the training loop before `train_step`. This prevents the TPU from beginning the backpropagation for the DiT model until the VAE forward pass completes.
**Recommendation:** Wrap the VAE `encode_fn` within the prefetching logic or push it down into the Grain data loader if possible. Alternatively, cache the VAE latents offline to avoid running the VAE at every epoch, completely removing this overhead and allowing the TPU to dedicate 100% of its FLOPs to DiT training.

### 3. Mixed Precision Completeness
**Issue:** `train_step` casts `latents` to `bfloat16` when `use_bf16=True`. However, `noise` is generated as `float32` by default in JAX, leading to implicit upcasting during the noise addition step (`noisy_latents = sqrt_alphas_cumprod * latents + sqrt_one_minus_alphas_cumprod * noise`). Additionally, without setting the precision policies on the linear layers of the DiT model, the TPU might still be computing matrix multiplications in `float32`.
**Recommendation:** 
- Explicitly cast `noise` to `bfloat16` during generation.
- Ensure that the Flax NNX layers are configured to use `bfloat16` compute types (using mixed precision policies or by manually casting module weights/inputs) so that the MXU (Matrix Multiply Unit) on the TPU runs optimally.

### 4. Sharding Strategy (Data Parallelism vs. FSDP)
**Issue:** The model parameters are currently replicated (`replicate_sharding`) across all devices. For a model with `hidden_size=384` and `depth=12`, this fits easily into TPU memory. However, if scaling up to larger architectures, pure data parallelism will hit memory limits due to duplicated optimizer states.
**Recommendation:** Implement Fully Sharded Data Parallelism (FSDP) using JAX's `PartitionSpec`. By sharding the model parameters, optimizer states, and gradients across the `data` axis, you can free up High Bandwidth Memory (HBM) and potentially increase the local batch size further for better MXU utilization.
