"""Diffusion sampling utilities."""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional

class DDIMSampler:
    """Simple DDIM sampler for DiT."""

    def __init__(self, num_train_timesteps: int = 1000):
        """Initialize the sampler."""
        self.num_train_timesteps = num_train_timesteps
        # Standard linear schedule
        self.betas = jnp.linspace(0.0001, 0.02, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)

    def sample(
        self,
        model_fn: Callable,
        shape: tuple,
        rng_key: jax.Array,
        num_inference_steps: int = 50,
        y: Optional[jax.Array] = None,
        null_y: Optional[jax.Array] = None,
        cfg_scale: float = 1.5
    ) -> jax.Array:
        """Sample from the model using DDIM.

        Args:
            model_fn: Function that takes (x, t, y) and returns predicted noise.
            shape: Shape of the noise to sample (N, H, W, C).
            rng_key: PRNGKey.
            num_inference_steps: Number of sampling steps.
            y: Labels for conditioning.
            null_y: Null labels for CFG.
            cfg_scale: Classifier-free guidance scale.

        Returns:
            Sampled latents.
        """
        key, subkey = jax.random.split(rng_key)
        x = jax.random.normal(subkey, shape)
        
        # DDIM sampling indices
        indices = jnp.linspace(self.num_train_timesteps - 1, 0, num_inference_steps).astype(jnp.int32)
        
        # Helper to get alpha_cumprod at t, or 1.0 if t < 0
        def get_alpha_cumprod(t):
            return jnp.where(t >= 0, self.alphas_cumprod[t], 1.0)

        for i in range(len(indices)):
            t_idx = indices[i]
            prev_t_idx = indices[i+1] if i+1 < len(indices) else -1
            
            t = jnp.full((shape[0],), t_idx)
            
            # Predict noise
            if cfg_scale > 1.0 and y is not None and null_y is not None:
                x_in = jnp.concatenate([x, x], axis=0)
                t_in = jnp.concatenate([t, t], axis=0)
                y_in = jnp.concatenate([y, null_y], axis=0)
                
                eps_all = model_fn(x_in, t_in, y_in)
                eps_cond, eps_uncond = jnp.split(eps_all, 2, axis=0)
                eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps = model_fn(x, t, y)
            
            alpha_t = get_alpha_cumprod(t_idx)
            alpha_prev = get_alpha_cumprod(prev_t_idx)
            
            # DDIM step (deterministic)
            # 1. Predict x0
            pred_x0 = (x - jnp.sqrt(1 - alpha_t) * eps) / jnp.sqrt(alpha_t)
            
            # 2. Direction pointing to xt
            dir_xt = jnp.sqrt(1 - alpha_prev) * eps
            
            # 3. x_prev
            x = jnp.sqrt(alpha_prev) * pred_x0 + dir_xt
            
        return x
