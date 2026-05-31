"""Unit tests for the DiT model."""

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from src.models.dit.dit import DiT

def test_dit_initialization():
    """Test that the DiT model can be initialized."""
    rngs = nnx.Rngs(0)
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_classes=10,
        rngs=rngs
    )
    assert isinstance(model, DiT)
    assert model.patch_size == 2
    assert len(model.blocks) == 2

def test_dit_forward_pass():
    """Test the forward pass of the DiT model."""
    rngs = nnx.Rngs(0)
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_classes=10,
        rngs=rngs
    )
    
    batch_size = 2
    x = jnp.zeros((batch_size, 32, 32, 4))
    t = jnp.array([0, 100])
    y = jnp.array([1, 5])
    
    output = model(x, t, y)
    
    # Check output shape. If learn_sigma is True (default), out_channels is in_channels * 2
    assert output.shape == (batch_size, 32, 32, 8)
