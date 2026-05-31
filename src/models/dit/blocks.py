"""Basic building blocks for the Diffusion Transformer (DiT)."""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Callable, Optional

def modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    """Modulate the input using scale and shift parameters.

    Args:
        x: Input tensor.
        shift: Modulation shift.
        scale: Modulation scale.

    Returns:
        Modulated tensor.
    """
    return x * (1 + scale.reshape(scale.shape[0], 1, scale.shape[1])) + shift.reshape(shift.shape[0], 1, shift.shape[1])

class DiTBlock(nnx.Module):
    """A single Diffusion Transformer block with AdaLN-Zero."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        rngs: Optional[nnx.Rngs] = None,
    ):
        """Initialize the DiT block.

        Args:
            hidden_size: Dimension of the embedding space.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden size to hidden_size.
            rngs: Random number generators.
        """
        self.norm1 = nnx.LayerNorm(num_features=hidden_size, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,
            out_features=hidden_size,
            rngs=rngs,
        )
        self.norm2 = nnx.LayerNorm(num_features=hidden_size, rngs=rngs)
        
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_hidden_size, rngs=rngs),
            nnx.gelu,
            nnx.Linear(mlp_hidden_size, hidden_size, rngs=rngs),
        )
        
        # AdaLN modulation parameters: 6 for each block 
        # (scale/shift for norm1, scale/shift for norm2, gate for attn, gate for mlp)
        # Critical: the final linear layer must be zero-initialized
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(hidden_size, 6 * hidden_size, kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros, rngs=rngs)
        )

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, D).
            c: Conditioning tensor of shape (B, D).

        Returns:
            Output tensor of shape (B, T, D).
        """
        # (B, 6*D)
        modulation = self.adaLN_modulation(c)
        # Split into 6 parts
        shift_ms = jnp.split(modulation, 6, axis=1)
        shift1, scale1, gate1, shift2, scale2, gate2 = shift_ms
        
        # Attention path
        h = modulate(self.norm1(x), shift1, scale1)
        h = self.attn(h, decode=False)
        x = x + gate1.reshape(gate1.shape[0], 1, gate1.shape[1]) * h
        
        # MLP path
        h = modulate(self.norm2(x), shift2, scale2)
        h = self.mlp(h)
        x = x + gate2.reshape(gate2.shape[0], 1, gate2.shape[1]) * h
        
        return x

class FinalLayer(nnx.Module):
    """The final layer of DiT for output projection."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, rngs: Optional[nnx.Rngs] = None):
        """Initialize the final layer.

        Args:
            hidden_size: Embedding dimension.
            patch_size: Resolution of each patch.
            out_channels: Number of output channels (e.g., 4 for latents).
            rngs: Random number generators.
        """
        self.norm_final = nnx.LayerNorm(num_features=hidden_size, rngs=rngs)
        self.linear = nnx.Linear(
            hidden_size, patch_size * patch_size * out_channels, 
            kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros, 
            rngs=rngs
        )
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                hidden_size, 2 * hidden_size, 
                kernel_init=jax.nn.initializers.zeros, bias_init=jax.nn.initializers.zeros, 
                rngs=rngs
            )
        )

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor.
            c: Conditioning tensor.

        Returns:
            Output projection.
        """
        modulation = self.adaLN_modulation(c)
        shift, scale = jnp.split(modulation, 2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
