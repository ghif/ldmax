"""Diffusion Transformer (DiT) model implementation."""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Optional, Tuple

from src.models.dit.blocks import DiTBlock, FinalLayer

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Create 2D sin-cos positional embeddings."""
    grid_h = jnp.arange(grid_size, dtype=jnp.float32)
    grid_w = jnp.arange(grid_size, dtype=jnp.float32)
    grid = jnp.meshgrid(grid_w, grid_h)
    grid = jnp.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = jnp.concatenate([emb_h, emb_w], axis=1)
    return pos_embed[None, :, :]

def get_1d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    grid = grid.flatten()
    out = jnp.einsum('m,d->md', grid, omega)
    emb = jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)
    return emb

class TimestepEmbedder(nnx.Module):
    """Embeds scalar timesteps into vector embeddings."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, rngs: Optional[nnx.Rngs] = None):
        """Initialize the timestep embedder."""
        self.mlp = nnx.Sequential(
            nnx.Linear(frequency_embedding_size, hidden_size, rngs=rngs),
            nnx.silu,
            nnx.Linear(hidden_size, hidden_size, rngs=rngs),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timesteps: jax.Array, dim: int, max_period: int = 10000) -> jax.Array:
        """Create sinusoidal timestep embeddings."""
        half_dim = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(0, half_dim) / half_dim
        )
        args = timesteps[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: jax.Array) -> jax.Array:
        """Forward pass."""
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nnx.Module):
    """Embeds class labels into vector embeddings."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float, rngs: Optional[nnx.Rngs] = None):
        """Initialize the label embedder."""
        use_cfg_embedding = True  # always reserved for null label
        self.embedding_table = nnx.Embed(num_classes + (1 if use_cfg_embedding else 0), hidden_size, rngs=rngs)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def __call__(self, labels: jax.Array, train: bool, rngs: Optional[nnx.Rngs] = None, force_drop_ids: Optional[jax.Array] = None) -> jax.Array:
        """Forward pass."""
        if train and self.dropout_prob > 0:
            if rngs is None:
                # Fallback to internal if not provided, though not ideal for nnx
                rng = nnx.Rngs().params()
            else:
                rng = rngs.params()
            # Randomly replace some labels with the null label (num_classes)
            drop_mask = jax.random.bernoulli(rng, self.dropout_prob, labels.shape)
            labels = jnp.where(drop_mask, self.num_classes, labels)
            
        if force_drop_ids is not None:
            labels = jnp.where(force_drop_ids, self.num_classes, labels)
            
        return self.embedding_table(labels)

class DiT(nnx.Module):
    """Diffusion Transformer."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        learn_sigma: bool = False,
        rngs: Optional[nnx.Rngs] = None,
    ):
        """Initialize DiT."""
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        
        # Patch embedding
        self.x_embedder = nnx.Linear(patch_size * patch_size * in_channels, hidden_size, rngs=rngs)
        
        # Timestep and label embedding
        self.t_embedder = TimestepEmbedder(hidden_size, rngs=rngs)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, dropout_prob=0.1, rngs=rngs)
        
        # Positional embedding (fixed 2D sin-cos)
        grid_size = input_size // patch_size
        pos_embed = get_2d_sincos_pos_embed(hidden_size, grid_size)
        self.pos_embed = nnx.Param(pos_embed)
        
        # DiT blocks
        self.blocks = nnx.List([
            DiTBlock(hidden_size, num_heads, mlp_ratio, rngs=rngs)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, rngs=rngs)

    def unpatchify(self, x: jax.Array) -> jax.Array:
        """Convert patched representation back to images."""
        p = self.patch_size
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        imgs = x.reshape(x.shape[0], h * p, w * p, c)
        return imgs

    def patchify(self, x: jax.Array) -> jax.Array:
        """Convert images to patches."""
        p = self.patch_size
        n, h, w, c = x.shape
        x = x.reshape(n, h // p, p, w // p, p, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        patches = x.reshape(n, (h // p) * (w // p), p * p * c)
        return patches

    def __call__(self, x: jax.Array, t: jax.Array, y: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        """Forward pass.
        
        Args:
            x: Input latents shape (N, H, W, C).
            t: Timesteps shape (N,).
            y: Labels shape (N,).
            rngs: Random number generators.
            
        Returns:
            Denoised latents shape (N, H, W, C).
        """
        # Patchify and embed
        x = self.patchify(x)
        x = self.x_embedder(x) + self.pos_embed
        
        # Embed conditioning
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, train=rngs is not None, rngs=rngs)
        c = t_emb + y_emb
        
        # DiT Blocks
        for block in self.blocks:
            x = block(x, c)
            
        # Final output
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
