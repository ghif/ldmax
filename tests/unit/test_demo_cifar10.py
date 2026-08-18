"""Tests for CIFAR-10 Gradio demo script."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


def test_cifar10_demo_class_names():
    from scripts.demo_cifar10 import CLASS_NAMES

    assert len(CLASS_NAMES) == 10
    assert CLASS_NAMES == [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]


def test_cifar10_to_images():
    from scripts.demo_cifar10 import _to_images

    samples = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)
    images = _to_images(samples)
    assert len(images) == 2
    assert images[0].shape == (32, 32, 3)
    assert images[0].dtype == np.uint8
    # 0.0 normalized -> 128 uint8
    assert np.all(images[0] == 128)


def test_cifar10_build_demo_model():
    pytest.importorskip("ml_collections")
    from scripts.demo_cifar10 import _build_demo_model
    from src.models.dit.dit import DiT
    from src.utils.config import load_config

    config = load_config("configs/cifar10_pixel.yaml")
    model = _build_demo_model(config, seed=42)
    assert isinstance(model, DiT)
    assert model.y_embedder.num_classes == 10


def test_cifar10_make_generate():
    pytest.importorskip("ml_collections")
    from scripts.demo_cifar10 import _make_generate
    from src.models.dit.dit import DiT
    from src.utils.config import load_config

    config = load_config("configs/cifar10_pixel.yaml")
    # Use a small model for test speed
    model = DiT(
        input_size=8,
        patch_size=2,
        in_channels=3,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_classes=10,
        label_mode="class",
        rngs=nnx.Rngs(0),
    )
    small_config = config
    small_config.model.input_size = 8
    small_config.model.in_channels = 3
    small_config.model.num_classes = 10

    generate = _make_generate(model, small_config)
    class_weights = [1.0] + [0.0] * 9
    images = generate(class_weights, num_samples=2, inference_steps=1, cfg_scale=1.5, seed=0)
    assert len(images) == 2
    assert images[0].shape == (8, 8, 3)
    assert images[0].dtype == np.uint8

    with pytest.raises(ValueError, match="positive influence"):
        generate([0.0] * 10, num_samples=2, inference_steps=1, cfg_scale=1.5, seed=0)


def test_cifar10_build_app(monkeypatch):
    pytest.importorskip("gradio")
    pytest.importorskip("ml_collections")
    from scripts.demo_cifar10 import build_app

    # Mock _restore_ema so it doesn't try downloading from GCS during test
    monkeypatch.setattr("scripts.demo_cifar10._restore_ema", lambda model, checkpoint: None)

    app = build_app("configs/cifar10_pixel.yaml", "fake_checkpoint", seed=0)
    assert app is not None
    assert app.title == "CIFAR-10 Diffusion Image Generator"
