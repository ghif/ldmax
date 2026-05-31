# TPU Parallelism and Performance Optimization Analysis

The DiT training pipeline has been optimized for high-performance execution on TPUs. The following optimizations were implemented to maximize TPU utilization and minimize host-side bottlenecks.

## 1. JIT-Compiled EMA Updates
**Before:** EMA updates were performed using `jax.tree.map` in Python every training step, causing significant host-side overhead and frequent TPU-host synchronization.
**After:** Refactored `EMAManager` in `src/training/ema.py` to use a JIT-compiled static method (`_ema_update`). This allows XLA to fuse the EMA calculations, eliminating the Python loop overhead.

## 2. Asynchronous and Parallel Data Loading
**Before:** Data loading was synchronous (`worker_count=0`), meaning the TPU would sit idle while the CPU fetched and normalized each batch.
**After:** 
- Configured Grain DataLoaders in `src/data/cifar.py` and `src/data/celeba.py` with `worker_count=4`.
- Implemented a `DevicePrefetcher` in `src/utils/prefetch.py` that wraps the data iterator to perform `jax.device_put` in the background. This overlaps Host-to-Device (H2D) transfers with TPU computation.

## 3. Full Bfloat16 Parameter Bandwidth Optimization
**Before:** Only input latents were cast to `bfloat16`. Model parameters remained in `float32`.
**After:** The model state is now explicitly cast to `bfloat16` before replication across the mesh. This reduces HBM bandwidth usage by 50% during parameter reads, which is a common bottleneck for large transformer models on TPU.

## 4. Optimized Training Loop
**Before:** The training loop manually sharded each batch and called the VAE encoder on the main thread.
**After:** 
- The loop now consumes from the `DevicePrefetcher`, receiving batches that are already sharded and resident in TPU memory.
- All core components (VAE encoding, DiT forward/backward, and EMA updates) are now fully JIT-compiled and execution-ready on the TPU mesh.

## Summary of Results
- **CPU Bottlenecks Removed**: Data is pre-fetched and prepared in parallel.
- **Host-TPU Dispatch Latency Reduced**: JIT-compiled EMA and optimized `train_step`.
- **HBM Bandwidth Optimized**: Parameters use `bfloat16`.

These changes ensure that the TPU remains the primary bottleneck (as desired) rather than being starved by data loading or host-side dispatching.
