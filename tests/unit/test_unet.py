"""Unit tests for the U-Net model."""

import jax
import jax.numpy as jnp
from flax import nnx
from src.models.unet.unet import UNetModel

def test_unet_initialization():
    """Test that the U-Net model can be initialized."""
    rngs = nnx.Rngs(0)
    model = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2),
        attention_resolutions=(2,),
        num_heads=8,
        rngs=rngs
    )
    assert isinstance(model, UNetModel)

def test_unet_forward_pass_class_cond():
    """Test the forward pass with class conditioning."""
    rngs = nnx.Rngs(0)
    model = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=64,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(2,),
        num_heads=4,
        num_classes=10,
        context_dim=64,
        rngs=rngs
    )
    
    batch_size = 2
    x = jnp.zeros((batch_size, 32, 32, 4))
    t = jnp.array([0, 100])
    y = jnp.array([1, 5])
    
    output = model(x, t, y)
    assert output.shape == (batch_size, 32, 32, 4)

def test_unet_forward_pass_attr_cond():
    """Test the forward pass with attribute conditioning."""
    rngs = nnx.Rngs(0)
    model = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=64,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(2,),
        num_heads=4,
        num_classes=40,
        label_mode="attributes",
        label_dim=40,
        context_dim=64,
        rngs=rngs
    )
    
    batch_size = 2
    x = jnp.zeros((batch_size, 32, 32, 4))
    t = jnp.array([0, 100])
    y = jnp.zeros((batch_size, 40))
    y = y.at[0, 31].set(1.0) # Smiling
    y = y.at[1, 20].set(1.0) # Male
    
    output = model(x, t, y)
    assert output.shape == (batch_size, 32, 32, 4)

def test_unet_shapes_multiple_resolutions():
    """Test that U-Net handles multiple downsampling levels correctly."""
    rngs = nnx.Rngs(0)
    model = UNetModel(
        in_channels=4,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2, 4),
        attention_resolutions=(4, 2),
        rngs=rngs
    )
    
    batch_size = 1
    x = jnp.zeros((batch_size, 32, 32, 4))
    t = jnp.array([0])
    
    output = model(x, t)
    assert output.shape == (batch_size, 32, 32, 4)
