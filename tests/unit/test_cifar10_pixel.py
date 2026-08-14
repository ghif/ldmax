"""Tests for the native-pixel CIFAR10 workflow."""

import jax.numpy as jnp
import pytest
from flax import nnx

from src.models.dit.dit import DiT, resolve_conditioning_mode
from src.training.sampler import DDIMSampler


def test_cifar10_pixel_model_outputs_rgb_class_conditioned():
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


def test_cifar10_pixel_config_validation():
    pytest.importorskip("datasets")
    pytest.importorskip("ml_collections")
    from src.training.cifar10_runner import _validate_config
    from src.utils.config import load_config

    config = load_config("configs/cifar10_pixel.yaml")
    _validate_config(config)
    config.model.input_size = 16
    with pytest.raises(ValueError, match="input_size=32"):
        _validate_config(config)


def test_ddim_pixel_clipping_keeps_clean_prediction_bounded():
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
