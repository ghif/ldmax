"""Unit tests for the DiT model."""

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from src.models.dit.dit import DiT, resolve_conditioning_mode

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
        learn_sigma=True,
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
        learn_sigma=True,
        rngs=rngs
    )
    
    batch_size = 2
    x = jnp.zeros((batch_size, 32, 32, 4))
    t = jnp.array([0, 100])
    y = jnp.array([1, 5])
    
    output = model(x, t, y)
    
    # Check output shape. If learn_sigma is True (default), out_channels is in_channels * 2
    assert output.shape == (batch_size, 32, 32, 8)

def test_dit_forward_pass_with_attribute_labels():
    """Test the forward pass with CelebA-style multi-attribute conditioning."""
    rngs = nnx.Rngs(0)
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=128,
        depth=2,
        num_heads=4,
        num_classes=40,
        label_mode="attributes",
        label_dim=40,
        learn_sigma=True,
        rngs=rngs
    )

    batch_size = 2
    x = jnp.zeros((batch_size, 32, 32, 4))
    t = jnp.array([0, 100])
    y = jnp.zeros((batch_size, 40), dtype=jnp.int32)
    y = y.at[0, 31].set(1)  # Smiling
    y = y.at[1, 20].set(1)  # Male

    output = model(x, t, y)
    assert output.shape == (batch_size, 32, 32, 8)


def test_conditioning_mode_validation():
    assert resolve_conditioning_mode("class") == "class"
    assert resolve_conditioning_mode("unconditional") == "none"
    with pytest.raises(ValueError, match="Expected 'class' or 'unconditional'"):
        resolve_conditioning_mode("invalid")


def test_unconditional_label_embedding_ignores_labels():
    model = DiT(
        input_size=28,
        patch_size=2,
        in_channels=1,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_classes=10,
        label_mode="none",
        label_dropout_prob=0.0,
        rngs=nnx.Rngs(0),
    )
    labels = jnp.array([0, 9], dtype=jnp.int32)
    embeddings = model.y_embedder(labels, train=False)
    assert embeddings.shape == (2, 32)
    assert jnp.all(embeddings == 0)


def test_class_conditioning_uses_label_embeddings():
    model = DiT(
        input_size=28,
        patch_size=2,
        in_channels=1,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_classes=10,
        label_mode="class",
        label_dropout_prob=0.0,
        rngs=nnx.Rngs(0),
    )
    labels = jnp.array([0, 9], dtype=jnp.int32)
    embeddings = model.y_embedder(labels, train=False)
    assert embeddings.shape == (2, 32)
    assert not jnp.allclose(embeddings[0], embeddings[1])
