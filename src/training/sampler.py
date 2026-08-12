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

    def sample_multi_conditional(
        self,
        model_fn: Callable,
        shape: tuple,
        rng_key: jax.Array,
        labels: jax.Array,
        null_label: int,
        weights: Optional[jax.Array] = None,
        num_inference_steps: int = 50,
        cfg_scale: float = 1.5,
    ) -> jax.Array:
        """Sample with equally weighted classifier-free class conditions.

        ``labels`` has shape ``(batch, num_conditions)``. The unconditional
        prediction is combined with the mean conditional direction:

        ``eps = eps_uncond + scale * sum(weights * (eps_cond - eps_uncond))``.

        This enables exploratory class blending with an existing checkpoint;
        it does not require retraining the model for multi-class outputs.
        """
        if labels.ndim != 2 or labels.shape[1] < 1:
            raise ValueError("labels must have shape (batch, num_conditions)")
        if labels.shape[0] != shape[0]:
            raise ValueError("labels batch dimension must match sample shape")
        if weights is None:
            weights = jnp.ones((labels.shape[1],), dtype=jnp.float32)
        weights = jnp.asarray(weights, dtype=jnp.float32)
        if weights.ndim != 1 or weights.shape[0] != labels.shape[1]:
            raise ValueError("weights must have one value per condition")
        if bool(jnp.any(weights < 0)) or bool(jnp.all(weights == 0)):
            raise ValueError("weights must be non-negative with at least one positive value")
        weights = weights / jnp.sum(weights)

        _, subkey = jax.random.split(rng_key)
        x = jax.random.normal(subkey, shape)
        indices = jnp.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps
        ).astype(jnp.int32)

        def get_alpha_cumprod(t):
            return jnp.where(t >= 0, self.alphas_cumprod[t], 1.0)

        batch_size, num_conditions = labels.shape
        for i in range(len(indices)):
            t_idx = indices[i]
            prev_t_idx = indices[i + 1] if i + 1 < len(indices) else -1
            t = jnp.full((batch_size,), t_idx)

            x_in = jnp.concatenate(
                [x] + [x] * num_conditions,
                axis=0,
            )
            t_in = jnp.tile(t, num_conditions + 1)
            condition_labels = labels.T.reshape(-1)
            y_in = jnp.concatenate(
                [jnp.full((batch_size,), null_label, dtype=labels.dtype), condition_labels],
                axis=0,
            )
            eps_all = model_fn(x_in, t_in, y_in)
            eps_uncond = eps_all[:batch_size]
            eps_cond = eps_all[batch_size:].reshape(num_conditions, batch_size, *eps_all.shape[1:])
            direction = jnp.sum(
                weights[:, None, None, None, None]
                * (eps_cond - eps_uncond[None, ...]),
                axis=0,
            )
            eps = eps_uncond + cfg_scale * direction

            alpha_t = get_alpha_cumprod(t_idx)
            alpha_prev = get_alpha_cumprod(prev_t_idx)
            pred_x0 = (x - jnp.sqrt(1 - alpha_t) * eps) / jnp.sqrt(alpha_t)
            direction = jnp.sqrt(1 - alpha_prev) * eps
            x = jnp.sqrt(alpha_prev) * pred_x0 + direction

        return x
