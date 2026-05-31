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
    key: jax.Array
) -> jax.Array:
    """Compute the diffusion loss for a batch of latents.

    Args:
        model: The DiT model.
        latents: Pre-encoded latent batch shape (B, H, W, C).
        labels: Class labels.
        key: PRNGKey for randomness.

    Returns:
        Scalar MSE loss.
    """
    # Split key for different operations
    noise_key, time_key = jax.random.split(key)
    
    # 1. Sample noise
    noise = jax.random.normal(noise_key, latents.shape)
    t = jax.random.randint(time_key, (latents.shape[0],), 0, 1000)
    
    # 2. Add noise to latents (forward diffusion)
    # Simple linear schedule
    sqrt_alpha_prod = jnp.cos(t / 1000 * jnp.pi / 2)[:, None, None, None]
    sqrt_one_minus_alpha_prod = jnp.sin(t / 1000 * jnp.pi / 2)[:, None, None, None]
    noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
    
    # 3. Predict noise with DiT
    model_output = model(noisy_latents, t, labels)
    
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
