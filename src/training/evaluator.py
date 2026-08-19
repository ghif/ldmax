"""Unified evaluation and sampling engine for pixel-space and latent-space diffusion."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from math import ceil, sqrt
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from PIL import Image

from src.data.factory import DatasetMetadata
from src.training.sampler import DDIMSampler
from src.utils.logging import TensorBoardLogger
from src.utils.vae import VAEManager


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get configuration attribute from Dict, ConfigDict, or Namespace."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if hasattr(obj, "get") and callable(obj.get):
        return obj.get(key, default)
    return getattr(obj, key, default)


def save_sample_grid(samples: jax.Array | np.ndarray, path: str | Path) -> None:
    """Save an NHWC batch of images in [0, 1] range as an image grid.

    Args:
        samples: Array of shape (N, H, W, C) with pixel values in [0.0, 1.0].
        path: File destination path for the saved image.
    """
    images = (np.asarray(samples).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    num_images = len(images)
    if num_images == 0:
        return

    columns = ceil(sqrt(num_images))
    rows = ceil(num_images / columns)
    height, width = images.shape[1:3]
    channels = images.shape[3] if images.ndim == 4 else 1

    mode = "L" if channels == 1 else "RGB"
    grid = Image.new(mode, (columns * width, rows * height))

    for index, image in enumerate(images):
        if channels == 1:
            img_slice = image[..., 0] if image.ndim == 3 else image
        else:
            img_slice = image[..., :3]
        grid.paste(
            Image.fromarray(img_slice, mode=mode),
            ((index % columns) * width, (index // columns) * height),
        )

    dest_path = Path(path).expanduser().resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(dest_path)


def unnormalize_pixels(samples: jax.Array) -> jax.Array:
    """Convert [-1.0, 1.0] diffusion outputs to [0.0, 1.0] image pixel values."""
    return jnp.clip((samples + 1.0) / 2.0, 0.0, 1.0)


@nnx.jit
def _model_forward(model: nnx.Module, x: jax.Array, t: jax.Array, y: jax.Array | None) -> jax.Array:
    """JIT-compiled model forward pass stripping learned sigma channels if present."""
    output = model(x, t, y)
    if output.shape[-1] == x.shape[-1] * 2:
        return jnp.split(output, 2, axis=-1)[0]
    return output


class Evaluator:
    """Unified evaluation and sampling engine for latent and raw pixel diffusion models."""

    def __init__(
        self,
        config: Any,
        metadata: DatasetMetadata,
        vae_manager: VAEManager | None = None,
        sampler: DDIMSampler | None = None,
    ):
        """Initialize the Evaluator.

        Args:
            config: Full experiment configuration.
            metadata: DatasetMetadata describing resolution, channels, and space.
            vae_manager: Optional VAEManager for latent decoding.
            sampler: Optional DDIMSampler (default is standard DDIMSampler).
        """
        self.config = config
        self.metadata = metadata
        self.sampler = sampler or DDIMSampler()

        if self.metadata.is_latent:
            self.vae_manager = vae_manager or VAEManager()
        else:
            self.vae_manager = None

    def decode_samples(self, samples: jax.Array) -> jax.Array:
        """Decode diffusion latents or unnormalize pixel samples to [0, 1] range."""
        if self.metadata.is_latent and self.vae_manager is not None:
            return self.vae_manager.decode(samples)
        return unnormalize_pixels(samples)

    def generate_samples(
        self,
        model: nnx.Module,
        batch: dict[str, Any],
        rng_key: jax.Array,
        sample_count: int | None = None,
        num_inference_steps: int | None = None,
        cfg_scale: float | None = None,
    ) -> jax.Array:
        """Generate visual samples from the given model using DDIM.

        Args:
            model: Instantiated diffusion model (or EMA replica).
            batch: Data batch providing conditioning labels.
            rng_key: PRNGKey for sampling noise.
            sample_count: Number of images to generate (default from config).
            num_inference_steps: DDIM denoising steps (default from config).
            cfg_scale: Classifier-free guidance scale (default from config).

        Returns:
            Decoded images in NHWC format within [0.0, 1.0].
        """
        eval_cfg = getattr(self.config, "evaluation", None)
        model_cfg = self.config.model

        if sample_count is None:
            sample_count = _cfg_get(eval_cfg, "sample_count", 16) if eval_cfg else 16
        if num_inference_steps is None:
            num_inference_steps = _cfg_get(eval_cfg, "num_inference_steps", 50) if eval_cfg else 50
        if cfg_scale is None:
            cfg_scale = _cfg_get(eval_cfg, "cfg_scale", 1.5) if eval_cfg else 1.5

        # Determine conditioning labels
        labels = batch.get("label")
        if labels is not None:
            count = min(sample_count, labels.shape[0])
            labels = labels[:count]
            if self.metadata.label_mode == "attributes":
                null_labels = jnp.zeros_like(labels)
            elif self.metadata.label_mode == "class":
                num_classes = _cfg_get(model_cfg, "num_classes", 10)
                null_labels = jnp.full_like(labels, num_classes)
            else:
                null_labels = None
                labels = None
        else:
            count = sample_count
            labels = None
            null_labels = None

        sample_shape = (
            count,
            model_cfg.input_size,
            model_cfg.input_size,
            model_cfg.in_channels,
        )

        def model_fn(x, t, y):
            return _model_forward(model, x, t, y)

        latents_or_pixels = self.sampler.sample(
            model_fn,
            sample_shape,
            rng_key,
            num_inference_steps=num_inference_steps,
            y=labels,
            null_y=null_labels,
            cfg_scale=cfg_scale,
        )

        return self.decode_samples(latents_or_pixels)

    def evaluate_and_log_samples(
        self,
        sampling_model: nnx.Module,
        ema_state: nnx.State,
        batch: dict[str, Any],
        rng_key: jax.Array,
        step: int,
        logger: TensorBoardLogger,
        output_dir: str | Path,
    ) -> tuple[jax.Array, float]:
        """Update sampling model from EMA, generate samples, log to TB, and save PNG grid.

        Args:
            sampling_model: Model instance dedicated to sampling.
            ema_state: Current EMA weights to synchronize.
            batch: Data batch providing conditioning labels.
            rng_key: PRNGKey.
            step: Current training step.
            logger: TensorBoard logger.
            output_dir: Experiment output directory.

        Returns:
            Tuple of (decoded_images, sampling_duration_seconds).
        """
        start_time = time.perf_counter()
        nnx.update(sampling_model, ema_state)

        images = self.generate_samples(
            model=sampling_model,
            batch=batch,
            rng_key=rng_key,
        )

        logger.log_images(step, "train/samples", images)
        logger.flush()

        sample_path = os.path.join(
            str(output_dir),
            "checkpoints",
            "samples",
            f"samples_step_{step:06d}.png",
        )
        save_sample_grid(images, sample_path)
        duration = time.perf_counter() - start_time
        return images, duration
