"""Unit tests for src/data/factory.py and src/models/factory.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from src.data.factory import create_dataloaders, get_dataset_metadata
from src.models.dit.dit import DiT
from src.models.factory import create_model
from src.utils.config import load_config


def test_dataset_metadata_cifar10():
    """Verify metadata extraction for CIFAR-10."""
    config = load_config("configs/cifar10_pixel.yaml")
    meta = get_dataset_metadata(config)
    assert meta.name == "cifar10"
    assert not meta.is_latent
    assert meta.image_size == 32
    assert meta.channels == 3
    assert meta.num_classes == 10
    assert meta.label_mode == "class"


def test_dataset_metadata_fashion_mnist():
    """Verify metadata extraction for Fashion MNIST."""
    config = load_config("configs/fashion_mnist.yaml")
    meta = get_dataset_metadata(config)
    assert meta.name == "fashion_mnist"
    assert not meta.is_latent
    assert meta.image_size == 28
    assert meta.channels == 1
    assert meta.num_classes == 10
    assert meta.label_mode == "class"


def test_dataset_metadata_celeba():
    """Verify metadata extraction for CelebA."""
    config = load_config("configs/celeba.yaml")
    meta = get_dataset_metadata(config)
    assert meta.name == "celeba"
    assert meta.is_latent
    assert meta.image_size == 256
    assert meta.channels == 3
    assert meta.num_classes == 40
    assert meta.label_mode == "attributes"
    assert meta.label_dim == 40


def test_create_model_dit():
    """Verify creation of DiT model from config."""
    config = load_config("configs/cifar10_pixel.yaml")
    config.model.hidden_size = 64
    config.model.depth = 2
    config.model.num_heads = 2
    config.training.use_bf16 = False

    rng_key = jnp.array([0, 1], dtype=jnp.uint32)
    model = create_model(config, rng_key)
    assert isinstance(model, DiT)
    assert model.in_channels == 3
    assert model.patch_size == 2


def test_create_model_unsupported():
    """Verify that unsupported model types raise a ValueError."""
    config = SimpleNamespace(
        model=SimpleNamespace(type="invalid_arch", in_channels=3),
        training=SimpleNamespace(use_bf16=False),
    )
    rng_key = jnp.array([0, 1], dtype=jnp.uint32)
    with pytest.raises(ValueError, match="Unsupported model type"):
        create_model(config, rng_key)


def test_create_dataloaders_mocked(monkeypatch):
    """Verify factory dispatches to correct loader functions."""
    mock_cifar_loader = MagicMock()
    monkeypatch.setattr("src.data.factory.get_cifar10_dataset", mock_cifar_loader)

    config = load_config("configs/cifar10_pixel.yaml")
    config.training.prefetch_size = 0
    bundle = create_dataloaders(config)

    assert bundle.metadata.name == "cifar10"
    assert mock_cifar_loader.call_count == 2
