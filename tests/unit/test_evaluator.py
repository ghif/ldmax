"""Unit tests for src/training/evaluator.py."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from src.data.factory import DatasetMetadata
from src.models.dit.dit import DiT
from src.training.evaluator import Evaluator, save_sample_grid, unnormalize_pixels


def test_save_sample_grid_grayscale(tmp_path):
    """Verify saving a grayscale 1-channel image grid."""
    samples = np.ones((4, 14, 14, 1), dtype=np.float32) * 0.5
    dest = tmp_path / "samples" / "gray_grid.png"
    save_sample_grid(samples, dest)

    assert dest.is_file()
    img = Image.open(dest)
    assert img.mode == "L"
    assert img.size == (28, 28)


def test_save_sample_grid_rgb(tmp_path):
    """Verify saving an RGB 3-channel image grid."""
    samples = np.ones((4, 16, 16, 3), dtype=np.float32) * 0.8
    dest = tmp_path / "samples" / "rgb_grid.png"
    save_sample_grid(samples, dest)

    assert dest.is_file()
    img = Image.open(dest)
    assert img.mode == "RGB"
    assert img.size == (32, 32)


def test_unnormalize_pixels():
    """Verify scaling from [-1, 1] to [0, 1]."""
    x = jnp.array([-1.0, 0.0, 1.0, -2.0, 2.0], dtype=jnp.float32)
    unnorm = unnormalize_pixels(x)
    assert jnp.allclose(unnorm, jnp.array([0.0, 0.5, 1.0, 0.0, 1.0]))


def test_evaluator_pixel_decode():
    """Verify pixel dataset decoding without VAE."""
    config = SimpleNamespace(
        model=SimpleNamespace(input_size=16, in_channels=1, num_classes=10),
        evaluation={"sample_count": 2, "num_inference_steps": 2, "cfg_scale": 1.0},
    )
    metadata = DatasetMetadata(
        name="fashion_mnist",
        is_latent=False,
        image_size=16,
        channels=1,
        num_classes=10,
        label_mode="class",
    )
    evaluator = Evaluator(config, metadata)
    assert evaluator.vae_manager is None

    dummy_samples = jnp.zeros((2, 16, 16, 1), dtype=jnp.float32)
    decoded = evaluator.decode_samples(dummy_samples)
    assert decoded.shape == (2, 16, 16, 1)
    assert jnp.allclose(decoded, 0.5)


def test_evaluator_generate_and_log(tmp_path):
    """Verify sample generation, tensorboard logging, and grid output with DiT."""
    config = SimpleNamespace(
        model=SimpleNamespace(
            input_size=8,
            patch_size=2,
            in_channels=1,
            hidden_size=32,
            depth=1,
            num_heads=2,
            num_classes=10,
            label_mode="class",
            label_dim=None,
            learn_sigma=False,
        ),
        evaluation={
            "sample_count": 2,
            "num_inference_steps": 2,
            "cfg_scale": 1.0,
        },
    )
    metadata = DatasetMetadata(
        name="fashion_mnist",
        is_latent=False,
        image_size=8,
        channels=1,
        num_classes=10,
        label_mode="class",
    )

    rngs = nnx.Rngs(0)
    model = DiT(
        input_size=8,
        patch_size=2,
        in_channels=1,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_classes=10,
        label_mode="class",
        rngs=rngs,
    )

    evaluator = Evaluator(config, metadata)
    batch = {"label": jnp.array([1, 2], dtype=jnp.int32)}
    rng_key = jax.random.key(42)

    logger = MagicMock()
    images, duration = evaluator.evaluate_and_log_samples(
        sampling_model=model,
        ema_state=nnx.state(model),
        batch=batch,
        rng_key=rng_key,
        step=10,
        logger=logger,
        output_dir=str(tmp_path),
    )

    assert images.shape == (2, 8, 8, 1)
    assert duration > 0
    assert logger.log_images.called
    assert (tmp_path / "checkpoints" / "samples" / "samples_step_000010.png").is_file()
