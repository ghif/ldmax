"""Training step implementation using JAX/Optax."""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Any, Dict, Tuple, Optional

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
    
    # 1. Sample noise
    # Ensure noise matches latents dtype for MXU efficiency
    noise = jax.random.normal(noise_key, latents.shape, dtype=latents.dtype)
    t = jax.random.randint(time_key, (latents.shape[0],), 0, 1000)
    
    # 2. Add noise to latents (forward diffusion)
    # Standard linear schedule
    betas = jnp.linspace(0.0001, 0.02, 1000)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas)
    
    sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - alphas_cumprod[t])[:, None, None, None]
    
    # Ensure constants match dtype
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.astype(latents.dtype)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.astype(latents.dtype)
    
    noisy_latents = sqrt_alphas_cumprod * latents + sqrt_one_minus_alphas_cumprod * noise
    
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
    return jnp.mean((pred_noise - noise) ** 2)

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
    if use_bf16:
        latents = latents.astype(jnp.bfloat16)

    def loss_fn(model):
        return compute_loss(model, latents, labels, rng_key)

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    
    return {"loss": loss}
