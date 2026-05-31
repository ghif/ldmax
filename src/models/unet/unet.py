"""Latent Diffusion U-Net model implementation."""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Any, Optional, Tuple, Sequence, List

from src.models.unet.blocks import ResBlock, SpatialTransformer, Upsample, Downsample

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
    """Embeds class labels or binary attribute vectors into embeddings."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float,
        label_mode: str = "class",
        label_dim: Optional[int] = None,
        rngs: Optional[nnx.Rngs] = None,
    ):
        """Initialize the label embedder."""
        self.label_mode = label_mode
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        if label_mode == "class":
            use_cfg_embedding = True  # always reserved for null label
            self.embedding_table = nnx.Embed(
                num_classes + (1 if use_cfg_embedding else 0),
                hidden_size,
                rngs=rngs,
            )
            self.label_dim = None
            self.proj = None
        elif label_mode == "attributes":
            if label_dim is None:
                raise ValueError("label_dim is required when label_mode='attributes'")
            self.label_dim = label_dim
            self.proj = nnx.Linear(label_dim, hidden_size, rngs=rngs)
            self.embedding_table = None
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

    def __call__(self, labels: jax.Array, train: bool, rngs: Optional[nnx.Rngs] = None, force_drop_ids: Optional[jax.Array] = None) -> jax.Array:
        """Forward pass."""
        if self.label_mode == "class":
            if train and self.dropout_prob > 0:
                if rngs is None:
                    rng = nnx.Rngs().params()
                else:
                    rng = rngs.params()
                drop_mask = jax.random.bernoulli(rng, self.dropout_prob, labels.shape)
                labels = jnp.where(drop_mask, self.num_classes, labels)

            if force_drop_ids is not None:
                labels = jnp.where(force_drop_ids, self.num_classes, labels)

            return self.embedding_table(labels)

        labels = labels.astype(jnp.float32)
        if train and self.dropout_prob > 0:
            if rngs is None:
                rng = nnx.Rngs().params()
            else:
                rng = rngs.params()
            drop_mask = jax.random.bernoulli(rng, self.dropout_prob, (labels.shape[0],))
            labels = jnp.where(drop_mask[:, None], jnp.zeros_like(labels), labels)

        if force_drop_ids is not None:
            labels = jnp.where(force_drop_ids[:, None], jnp.zeros_like(labels), labels)

        return self.proj(labels)

class UNetModel(nnx.Module):
    """Latent Diffusion U-Net model."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        attention_resolutions: Sequence[int] = (4, 2, 1),
        num_res_blocks: int = 2,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        num_heads: int = 8,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        num_classes: int = 1000,
        label_mode: str = "class",
        label_dim: Optional[int] = None,
        use_spatial_transformer: bool = True,
        dropout: float = 0.0,
        rngs: Optional[nnx.Rngs] = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.attention_resolutions = attention_resolutions
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        
        time_embed_dim = model_channels * 4
        self.time_embed = TimestepEmbedder(time_embed_dim, rngs=rngs)
        
        self.context_dim = context_dim
        if context_dim is not None:
            self.label_embedder = LabelEmbedder(
                num_classes,
                context_dim,
                dropout_prob=0.1,
                label_mode=label_mode,
                label_dim=label_dim,
                rngs=rngs,
            )
        else:
            self.label_embedder = None
        
        self.input_blocks = nnx.List([])
        self.input_blocks.append(nnx.Conv(in_channels, model_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs))
        
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = []
                layers.append(ResBlock(ch, mult * model_channels, emb_channels=time_embed_dim, dropout=dropout, rngs=rngs))
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, num_heads, ch // num_heads, depth=transformer_depth, context_dim=context_dim, rngs=rngs))
                self.input_blocks.append(nnx.Sequential(*layers))
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch, use_conv=True, rngs=rngs))
                input_block_chans.append(ch)
                ds *= 2
                
        self.middle_block = nnx.Sequential(
            ResBlock(ch, ch, emb_channels=time_embed_dim, dropout=dropout, rngs=rngs),
            SpatialTransformer(ch, num_heads, ch // num_heads, depth=transformer_depth, context_dim=context_dim, rngs=rngs),
            ResBlock(ch, ch, emb_channels=time_embed_dim, dropout=dropout, rngs=rngs),
        )
        
        self.output_blocks = nnx.List([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = []
                ich = input_block_chans.pop()
                layers.append(ResBlock(ch + ich, mult * model_channels, emb_channels=time_embed_dim, dropout=dropout, rngs=rngs))
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(SpatialTransformer(ch, num_heads, ch // num_heads, depth=transformer_depth, context_dim=context_dim, rngs=rngs))
                
                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch, use_conv=True, rngs=rngs))
                    ds //= 2
                
                self.output_blocks.append(nnx.Sequential(*layers))
                
        self.out = nnx.Sequential(
            nnx.GroupNorm(num_groups=32, num_features=ch, rngs=rngs),
            nnx.silu,
            nnx.Conv(ch, out_channels, kernel_size=(3, 3), padding="SAME", kernel_init=jax.nn.initializers.zeros, rngs=rngs),
        )

    def __call__(self, x: jax.Array, timesteps: jax.Array, y: Optional[jax.Array] = None, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        """Forward pass.
        
        Args:
            x: Input feature maps shape (N, H, W, C).
            timesteps: Scalar timesteps shape (N,).
            y: Labels or attribute vectors.
            rngs: Random number generators.
            
        Returns:
            Output feature maps shape (N, H, W, C).
        """
        emb = self.time_embed(timesteps)
        
        context = None
        if self.label_embedder is not None and y is not None:
            context = self.label_embedder(y, train=rngs is not None, rngs=rngs)
            # Reshape for cross-attention context (N, 1, D)
            context = context[:, None, :]
        
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, nnx.Sequential):
                for layer in module.layers:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, SpatialTransformer):
                        h = layer(h, context)
                    else:
                        h = layer(h)
            elif isinstance(module, Downsample):
                h = module(h)
            else:
                h = module(h)
            hs.append(h)
            
        # Middle block
        for layer in self.middle_block.layers:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            elif isinstance(layer, SpatialTransformer):
                h = layer(h, context)
            else:
                h = layer(h)
                
        # Decoder
        for module in self.output_blocks:
            h = jnp.concatenate([h, hs.pop()], axis=-1)
            if isinstance(module, nnx.Sequential):
                for layer in module.layers:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, SpatialTransformer):
                        h = layer(h, context)
                    else:
                        h = layer(h)
            else:
                h = module(h)
                
        return self.out(h)
