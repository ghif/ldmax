"""VAE utilities for encoding/decoding using diffusers."""

import jax
import jax.numpy as jnp
from diffusers import FlaxAutoencoderKL
from typing import Tuple, Optional

class VAEManager:
    """Manages the pre-trained VAE for latent diffusion.
    
    This class wraps Hugging Face's FlaxAutoencoderKL to provide an easy interface
    for moving between pixel space [-1, 1] and latent space.
    """

    def __init__(
        self, 
        model_id: str = "enterprise-explorers/sd-vae-ft-mse-flax", 
        subfolder: Optional[str] = None, 
        dtype: jnp.dtype = jnp.float32
    ):
        """Initialize and load the pre-trained VAE.

        Args:
            model_id: Hugging Face model ID.
            subfolder: Subfolder in the repository containing the VAE.
            dtype: JAX data type for the VAE weights.
        """
        self.model_id = model_id
        self.dtype = dtype
        
        # Load VAE. Note: enterprise-explorers/sd-vae-ft-mse-flax doesn't use a subfolder.
        self.model, self.params = FlaxAutoencoderKL.from_pretrained(
            model_id, 
            subfolder=subfolder, 
            dtype=dtype
        )
        # The scaling factor used in Stable Diffusion to scale latents to unit variance.
        self.scaling_factor = 0.18215

    def encode(self, images: jax.Array, key: jax.Array, params: Optional[dict] = None) -> jax.Array:
        """Encode images into latent representations.

        Args:
            images: Image tensor of shape (B, H, W, 3) in range [-1, 1].
            key: PRNGKey for sampling from the latent distribution.
            params: Optional VAE parameters. If None, uses self.params.

        Returns:
            Latent tensor of shape (B, H/8, W/8, 4).
        """
        if params is None:
            params = self.params
            
        # HF FlaxAutoencoderKL expects NCHW
        if images.shape[-1] == 3:
            images = jnp.transpose(images, (0, 3, 1, 2))
        
        # Encode to latent distribution
        latent_dist = self.model.apply({"params": params}, images, method=self.model.encode).latent_dist
        
        # Sample from the distribution (reparameterization trick)
        latents = latent_dist.sample(key)
        
        # Scale latents
        # Note: FlaxAutoencoderKL usually returns NHWC if images were NCHW? 
        # Actually, based on empirical trace, it returns NHWC (B, H, W, C).
        return latents * self.scaling_factor

    def decode(self, latents: jax.Array, params: Optional[dict] = None) -> jax.Array:
        """Decode latents back into images.

        Args:
            latents: Latent tensor of shape (B, h, w, 4).
            params: Optional VAE parameters. If None, uses self.params.

        Returns:
            Image tensor of shape (B, H, W, 3) in range [0, 1].
        """
        if params is None:
            params = self.params
            
        # Unscale latents
        latents = latents / self.scaling_factor
        
        # Decode to pixel space. 
        # Note: If encode returns NHWC, decode likely expects NHWC.
        image_out = self.model.apply({"params": params}, latents, method=self.model.decode).sample
        
        # Post-process: NCHW -> NHWC, then [-1, 1] -> [0, 1]
        # If image_out is already NHWC, this will fail or be wrong.
        # But most diffusers models return NCHW.
        if image_out.shape[1] == 3:
            image_out = jnp.transpose(image_out, (0, 2, 3, 1))
            
        image_out = (image_out / 2 + 0.5).clip(0, 1)
        
        return image_out
