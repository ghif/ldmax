"""Diffusion Transformer (DiT) model implementation."""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Optional, Tuple

from src.models.dit.blocks import DiTBlock, FinalLayer

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

    def __call__(self, labels: jax.Array, train: bool, force_drop_ids: Optional[jax.Array] = None) -> jax.Array:
        """Forward pass."""
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            # Implementation of classifier-free guidance dropout (simplified)
            # In a real impl, we would replace labels with num_classes id
            pass
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
        learn_sigma: bool = True,
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
        
        # Positional embedding (learnable for simplicity, but DiT uses 2D sin-cos)
        num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nnx.Param(jnp.zeros((1, num_patches, hidden_size)))
        
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
        x = jnp.einsum('nhwpqc->nhpwqc', x)
        imgs = x.reshape(x.shape[0], h * p, w * p, c)
        return imgs

    def patchify(self, x: jax.Array) -> jax.Array:
        """Convert images to patches."""
        p = self.patch_size
        n, h, w, c = x.shape
        x = x.reshape(n, h // p, p, w // p, p, c)
        x = jnp.einsum('nhpwqc->nhw pqc', x)
        patches = x.reshape(n, (h // p) * (w // p), p * p * c)
        return patches

    def __call__(self, x: jax.Array, t: jax.Array, y: jax.Array) -> jax.Array:
        """Forward pass.
        
        Args:
            x: Input latents shape (N, H, W, C).
            t: Timesteps shape (N,).
            y: Labels shape (N,).
            
        Returns:
            Denoised latents shape (N, H, W, C).
        """
        # Patchify and embed
        x = self.patchify(x)
        x = self.x_embedder(x) + self.pos_embed
        
        # Embed conditioning
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, train=True)
        c = t_emb + y_emb
        
        # DiT Blocks
        for block in self.blocks:
            x = block(x, c)
            
        # Final output
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x
