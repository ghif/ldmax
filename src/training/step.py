"""Training step implementation using JAX/Optax."""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

NUM_TRAIN_TIMESTEPS = 1000
BETAS = np.linspace(0.0001, 0.02, NUM_TRAIN_TIMESTEPS)
ALPHAS_CUMPROD = np.cumprod(1.0 - BETAS)


def compute_loss(
    model: Any,
    latents: jax.Array,
    labels: jax.Array,
    key: jax.Array,
    train: bool = True,
) -> jax.Array:
    """Compute the diffusion loss for a batch of latents.

    Args:
        model: The DiT model.
        latents: Pre-encoded latent batch shape (B, H, W, C).
        labels: Class labels.
        key: PRNGKey for randomness.
        train: Whether to enable classifier-free label dropout.

    Returns:
        Scalar MSE loss.
    """
    # Split key for different operations
    noise_key, time_key, model_key = jax.random.split(key, 3)

    # Keep the diffusion target and schedule in FP32 for stable loss scaling.
    noise = jax.random.normal(noise_key, latents.shape, dtype=jnp.float32)
    latents_fp32 = latents.astype(jnp.float32)
    t = jax.random.randint(time_key, (latents.shape[0],), 0, NUM_TRAIN_TIMESTEPS)

    # 2. Add noise to latents (forward diffusion)
    # Standard linear schedule
    alphas_cumprod = jnp.asarray(ALPHAS_CUMPROD, dtype=jnp.float32)
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod[t])[:, None, None, None]

    # Ensure constants match dtype
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.astype(jnp.float32)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.astype(jnp.float32)

    noisy_latents = sqrt_alphas_cumprod * latents_fp32 + sqrt_one_minus_alphas_cumprod * noise

    # 3. Predict noise with DiT
    model_output = model(
        noisy_latents,
        t,
        labels,
        rngs=nnx.Rngs(model_key) if train else None,
    )

    # If learn_sigma is True, model_output has 2*C channels.
    # We take the first C channels for noise prediction.
    if model_output.shape[-1] == latents.shape[-1] * 2:
        pred_noise, _ = jnp.split(model_output, 2, axis=-1)
    else:
        pred_noise = model_output

    # 4. MSE Loss
    return jnp.mean((pred_noise.astype(jnp.float32) - noise) ** 2, dtype=jnp.float32)

@nnx.jit(static_argnames="use_bf16")
def train_step(
    model: Any,
    optimizer: nnx.Optimizer,
    latents: jax.Array,
    labels: jax.Array,
    rng_key: jax.Array,
    use_bf16: bool = False
) -> Dict[str, jax.Array]:
    """Perform a single training step on latents.

    Args:
        model: The DiT model.
        optimizer: The nnx.Optimizer instance.
        latents: Latent batch.
        labels: Class labels.
        rng_key: Raw JAX PRNGKey.
        use_bf16: Whether to use bfloat16 mixed precision.

    Returns:
        Dictionary of metrics (e.g., loss).
    """
    def loss_fn(model):
        return compute_loss(model, latents, labels, rng_key)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return {"loss": loss}
