"""Tests for the native-pixel CIFAR10 workflow."""

import jax.numpy as jnp
from flax import nnx

from src.models.dit.dit import DiT, resolve_conditioning_mode
from src.training.sampler import DDIMSampler


def test_cifar10_pixel_model_outputs_rgb_class_conditioned():
    """Verify class-conditioned DiT output shape for pixel CIFAR-10."""
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=2,
        num_classes=10,
        label_mode="class",
        rngs=nnx.Rngs(0),
    )
    output = model(jnp.zeros((2, 32, 32, 3)), jnp.array([0, 100]), jnp.array([1, 5]))
    assert output.shape == (2, 32, 32, 3)


def test_cifar10_pixel_model_outputs_rgb_unconditionally():
    """Verify unconditional DiT output shape for pixel CIFAR-10."""
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=64,
        depth=2,
        num_heads=2,
        num_classes=10,
        label_mode=resolve_conditioning_mode("unconditional"),
        rngs=nnx.Rngs(0),
    )
    output = model(jnp.zeros((2, 32, 32, 3)), jnp.array([0, 100]), jnp.array([1, 5]))
    assert output.shape == (2, 32, 32, 3)


def test_ddim_pixel_clipping_keeps_clean_prediction_bounded():
    """Verify that DDIM clip_denoised keeps outputs within [-1.0, 1.0]."""
    sampler = DDIMSampler()

    def model_fn(x, t, y):
        del t, y
        return jnp.zeros_like(x)

    # A one-step transition with zero predicted noise starts from a large
    # value, so clipping is observable through the final clean sample.
    result = sampler.sample(
        model_fn,
        (1, 4, 4, 3),
        jnp.array([0, 1], dtype=jnp.uint32),
        num_inference_steps=1,
        y=jnp.array([0]),
        clip_denoised=True,
    )
    assert result.shape == (1, 4, 4, 3)
    assert jnp.all(result <= 1.0)
    assert jnp.all(result >= -1.0)
