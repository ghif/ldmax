# Data Model: JAX DiT Training Pipeline

## Entities

### 1. DiT Model (Flax NNX Module)
The core transformer-based neural network.
- **Attributes**:
    - `depth` (int): Number of transformer layers.
    - `hidden_size` (int): Dimension of the embedding space.
    - `num_heads` (int): Number of attention heads.
    - `patch_size` (int): Size of the patches (typically 2 for CIFAR).
    - `num_classes` (int): Number of conditioning classes (10 for CIFAR-10).
- **State**:
    - `params` (nnx.Param): Learnable weights and biases.
    - `ema_params` (nnx.Variable): Exponentially averaged weights (tracked via `nnx.EMA`).

### 2. Training State (Unified Object)
Managed via a combination of Flax NNX and Orbax for checkpointing.
- **Components**:
    - `step` (int): Current training iteration.
    - `model`: Instance of the DiT model.
    - `optimizer`: `nnx.Optimizer` instance wrapping an Optax transformation.
    - `ema`: `nnx.EMA` instance tracking the model parameters.

### 3. Data Record (Grain Batch)
- **Image Batch**: `jnp.ndarray` (f32) shape `(B, 32, 32, 3)` in range `[-1, 1]`.
- **Latent Batch**: `jnp.ndarray` (f32/bf16) shape `(B, 4, 4, 4)` after VAE encoding.
- **Label Batch**: `jnp.ndarray` (i32) shape `(B,)` for class conditioning.

### 4. Configuration Schema
Stored in YAML and parsed into a nested structure (e.g., using `ml_collections` or `omegaconf`).
- **Model Config**: `depth`, `hidden_size`, `num_heads`, `patch_size`.
- **Training Config**: `learning_rate`, `batch_size`, `total_steps`, `ema_decay`, `mixed_precision` (bool).
- **Evaluation Config**: `sampling_interval`, `fid_interval`, `num_samples`.

## Data Flows

1. **Preprocessing (Grain)**: Raw Image -> Normalize -> Batch -> Training Loop.
2. **Encoding (JAX)**: Image Batch -> VAE Encoder -> Latent Batch (inside `train_step`).
3. **Forward Pass**: Latent Batch + Label Batch + Timestep -> DiT Model -> Noise Prediction.
4. **Backward Pass**: Loss calculation -> Grad computation -> Optax Update -> Master Weights Update -> EMA Update.
5. **Evaluation**: EMA Weights -> Sampler -> Reconstructed Image -> FID calculation.
