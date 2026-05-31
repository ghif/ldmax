"""Building blocks for the U-Net architecture."""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Callable, Optional, Tuple, Sequence

def zero_module(module):
    """Zero out the parameters of a module and return it."""
    # In NNX, we can't easily zero out after init without manual state manipulation.
    # We prefer using jax.nn.initializers.zeros in the constructors.
    return module

class Identity(nnx.Module):
    """A module that returns the input as is."""
    def __call__(self, x: jax.Array) -> jax.Array:
        return x

class Upsample(nnx.Module):
    """Convolutional upsampling layer."""

    def __init__(self, channels: int, use_conv: bool, rngs: nnx.Rngs):
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        b, h, w, c = x.shape
        x = jax.image.resize(x, (b, h * 2, w * 2, c), method="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nnx.Module):
    """Convolutional downsampling layer."""

    def __init__(self, channels: int, use_conv: bool, rngs: nnx.Rngs):
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nnx.Conv(channels, channels, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs)
        else:
            self.avg_pool = lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_conv:
            return self.conv(x)
        else:
            return nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

class ResBlock(nnx.Module):
    """Residual block with time-embedding conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        emb_channels: int = 512,
        dropout: float = 0.0,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.out_channels = out_channels or in_channels
        
        self.in_layers = nnx.Sequential(
            nnx.GroupNorm(num_groups=32, num_features=in_channels, rngs=rngs),
            nnx.silu,
            nnx.Conv(in_channels, self.out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs),
        )
        
        self.emb_layers = nnx.Sequential(
            nnx.silu,
            nnx.Linear(emb_channels, self.out_channels, rngs=rngs),
        )
        
        self.out_layers = nnx.Sequential(
            nnx.GroupNorm(num_groups=32, num_features=self.out_channels, rngs=rngs),
            nnx.silu,
            nnx.Dropout(dropout, rngs=rngs),
            nnx.Conv(self.out_channels, self.out_channels, kernel_size=(3, 3), padding="SAME", kernel_init=jax.nn.initializers.zeros, rngs=rngs),
        )
        
        if self.out_channels != in_channels:
            self.skip_connection = nnx.Conv(in_channels, self.out_channels, kernel_size=(1, 1), padding="SAME", rngs=rngs)
        else:
            self.skip_connection = Identity()

    def __call__(self, x: jax.Array, emb: jax.Array) -> jax.Array:
        h = self.in_layers(x)
        
        # Inject time embedding
        emb_out = self.emb_layers(emb)
        h = h + emb_out[:, None, None, :]
        
        h = self.out_layers(h)
        return self.skip_connection(x) + h

class CrossAttention(nnx.Module):
    """Multi-head cross-attention mechanism."""

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        rngs: Optional[nnx.Rngs] = None,
    ):
        inner_dim = num_heads * head_dim
        context_dim = context_dim or query_dim
        
        self.scale = head_dim**-0.5
        self.num_heads = num_heads
        
        self.to_q = nnx.Linear(query_dim, inner_dim, use_bias=False, rngs=rngs)
        self.to_k = nnx.Linear(context_dim, inner_dim, use_bias=False, rngs=rngs)
        self.to_v = nnx.Linear(context_dim, inner_dim, use_bias=False, rngs=rngs)
        
        self.to_out = nnx.Sequential(
            nnx.Linear(inner_dim, query_dim, rngs=rngs),
            nnx.Dropout(0.0, rngs=rngs),
        )

    def __call__(self, x: jax.Array, context: Optional[jax.Array] = None) -> jax.Array:
        b, t, d = x.shape
        context = context if context is not None else x
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.reshape(b, t, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = k.reshape(b, k.shape[1], self.num_heads, -1).transpose(0, 2, 1, 3)
        v = v.reshape(b, v.shape[1], self.num_heads, -1).transpose(0, 2, 1, 3)
        
        # Attention
        sim = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = jax.nn.softmax(sim, axis=-1)
        
        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        
        return self.to_out(out)

class BasicTransformerBlock(nnx.Module):
    """A single transformer block with self and cross attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_dim: Optional[int] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn1 = CrossAttention(dim, num_heads=num_heads, head_dim=head_dim, rngs=rngs) # Self-attn
        
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn2 = CrossAttention(dim, context_dim=context_dim, num_heads=num_heads, head_dim=head_dim, rngs=rngs) # Cross-attn
        
        self.norm3 = nnx.LayerNorm(dim, rngs=rngs)
        self.ff = nnx.Sequential(
            nnx.Linear(dim, dim * 4, rngs=rngs),
            nnx.gelu,
            nnx.Linear(dim * 4, dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array, context: Optional[jax.Array] = None) -> jax.Array:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x

class SpatialTransformer(nnx.Module):
    """Transformer block for spatial feature maps."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        depth: int = 1,
        context_dim: Optional[int] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.norm = nnx.GroupNorm(num_groups=32, num_features=in_channels, rngs=rngs)
        self.proj_in = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), padding="SAME", rngs=rngs)
        
        self.transformer_blocks = nnx.List([
            BasicTransformerBlock(in_channels, num_heads, head_dim, context_dim, rngs=rngs)
            for _ in range(depth)
        ])
        
        self.proj_out = nnx.Conv(in_channels, in_channels, kernel_size=(1, 1), padding="SAME", kernel_init=jax.nn.initializers.zeros, rngs=rngs)

    def __call__(self, x: jax.Array, context: Optional[jax.Array] = None) -> jax.Array:
        b, h, w, c = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        # Flatten spatial dimensions
        x = x.reshape(b, h * w, c)
        
        for block in self.transformer_blocks:
            x = block(x, context=context)
            
        # Reshape back to spatial dimensions
        x = x.reshape(b, h, w, c)
        
        return x_in + self.proj_out(x)
