"""Tests for unified Gradio demo script."""

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx


def test_demo_class_names():
    """Test CIFAR-10, Fashion MNIST, and CelebA attribute lists."""
    from scripts.demo import CELEBA_ATTRIBUTE_NAMES, CIFAR10_CLASSES, FASHION_CLASSES

    assert len(CIFAR10_CLASSES) == 10
    assert len(FASHION_CLASSES) == 10
    assert len(CELEBA_ATTRIBUTE_NAMES) == 40
    assert CIFAR10_CLASSES[0] == "airplane"
    assert FASHION_CLASSES[0] == "T-shirt/top"
    assert "Smiling" in CELEBA_ATTRIBUTE_NAMES


def test_demo_to_rgb_images():
    """Test conversion of tensor samples to uint8 RGB images."""
    from scripts.demo import _to_rgb_images

    samples = jnp.zeros((2, 32, 32, 3), dtype=jnp.float32)
    images = _to_rgb_images(samples)
    assert len(images) == 2
    assert images[0].shape == (32, 32, 3)
    assert images[0].dtype == np.uint8
    assert np.all(images[0] == 128)


def test_demo_to_grayscale_images():
    """Test conversion of tensor samples to uint8 grayscale images."""
    from scripts.demo import _to_grayscale_images

    samples = jnp.zeros((2, 28, 28, 1), dtype=jnp.float32)
    images = _to_grayscale_images(samples)
    assert len(images) == 2
    assert images[0].shape == (28, 28)
    assert images[0].dtype == np.uint8
    assert np.all(images[0] == 128)


def test_demo_build_demo_model():
    """Test building demo DiT models for CIFAR-10, Fashion MNIST, and CelebA configs."""
    pytest.importorskip("ml_collections")
    from scripts.demo import _build_demo_model
    from src.models.dit.dit import DiT
    from src.utils.config import load_config

    cifar_config = load_config("configs/cifar10_pixel.yaml")
    cifar_model = _build_demo_model(cifar_config, seed=42)
    assert isinstance(cifar_model, DiT)
    assert cifar_model.y_embedder.num_classes == 10

    fashion_config = load_config("configs/fashion_mnist_tpu_v4.yaml")
    fashion_model = _build_demo_model(fashion_config, seed=42)
    assert isinstance(fashion_model, DiT)
    assert fashion_model.y_embedder.num_classes == 10

    celeba_config = load_config("configs/celeba.yaml")
    celeba_model = _build_demo_model(celeba_config, seed=42)
    assert isinstance(celeba_model, DiT)
    assert cifar_model.y_embedder is not None


def test_demo_make_generate():
    """Test sampling generator closure with multi-conditional weights."""
    pytest.importorskip("ml_collections")
    from scripts.demo import _make_generate
    from src.models.dit.dit import DiT
    from src.utils.config import load_config

    config = load_config("configs/cifar10_pixel.yaml")
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

    generate_rgb = _make_generate(model, small_config, is_grayscale=False)
    class_weights = [1.0] + [0.0] * 9
    images = generate_rgb(
        class_weights, num_samples=2, inference_steps=1, cfg_scale=1.5, seed=0
    )
    assert len(images) == 2
    assert images[0].shape == (8, 8, 3)
    assert images[0].dtype == np.uint8

    with pytest.raises(ValueError, match="positive influence"):
        generate_rgb([0.0] * 10, num_samples=2, inference_steps=1, cfg_scale=1.5, seed=0)


def test_demo_make_celeba_generate():
    """Test CelebA latent sampling and mock VAE decoding."""
    pytest.importorskip("ml_collections")
    from scripts.demo import _make_celeba_generate
    from src.models.dit.dit import DiT
    from src.utils.config import load_config

    config = load_config("configs/celeba.yaml")
    model = DiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_classes=40,
        label_mode="attributes",
        label_dim=40,
        rngs=nnx.Rngs(0),
    )
    small_config = config
    small_config.model.input_size = 8
    small_config.model.in_channels = 4
    small_config.model.num_classes = 40
    small_config.model.label_dim = 40

    class MockVAE:
        def decode(self, latents):
            b = latents.shape[0]
            return jnp.ones((b, 64, 64, 3), dtype=jnp.float32)

    celeba_gen = _make_celeba_generate(model, small_config, vae_manager=MockVAE())
    images = celeba_gen(
        ["Smiling", "Young"], num_samples=2, inference_steps=1, cfg_scale=4.0, seed=0
    )
    assert len(images) == 2
    assert images[0].shape == (64, 64, 3)
    assert images[0].dtype == np.uint8
    assert np.all(images[0] == 255)


def test_demo_build_app(monkeypatch):
    """Test building the complete Gradio app with mocked checkpoint restoration."""
    pytest.importorskip("gradio")
    pytest.importorskip("ml_collections")
    from scripts.demo import build_app

    monkeypatch.setattr(
        "scripts.demo._restore_model_ema", lambda model, checkpoint: None
    )

    class MockVAE:
        def decode(self, latents):
            b = latents.shape[0]
            return jnp.ones((b, 64, 64, 3), dtype=jnp.float32)

    app = build_app(
        cifar10_config_path="configs/cifar10_pixel.yaml",
        cifar10_checkpoint="fake_cifar_cp",
        fashion_config_path="configs/fashion_mnist_tpu_v4.yaml",
        fashion_checkpoint="fake_fashion_cp",
        celeba_config_path="configs/celeba.yaml",
        celeba_checkpoint="fake_celeba_cp",
        vae_manager=MockVAE(),
        seed=0,
    )
    assert app is not None
    assert app.title == "LDMAX Diffusion Image Generator"
