"""Unified standalone image generator for DiT checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from src.data.celeba import CELEBA_ATTRIBUTE_NAMES
from src.data.factory import get_dataset_metadata
from src.models.factory import create_model
from src.training.checkpointing import (
    resolve_resume_checkpoint,
    restore_args,
    restore_nnx_state,
    restore_template,
    validate_nnx_state,
)
from src.training.evaluator import Evaluator, save_sample_grid
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.rng import RNGManager


def generate_samples(
    config: str | Path | Any,
    checkpoint: str | Path = "",
    num_samples: int = 16,
    num_inference_steps: int = 50,
    cfg_scale: float = 1.5,
    class_id: int = -1,
    attribute_names: str = "Smiling",
    seed: int = 42,
    output_path: str | Path = "./samples.png",
    use_ema: bool = True,
) -> jax.Array:
    """Generate image samples from a trained checkpoint or initialized model.

    Args:
        config: Path to YAML config file or loaded config.
        checkpoint: Optional path to Orbax checkpoint (e.g. 'outputs/cifar/checkpoints/5000').
        num_samples: Number of samples to generate.
        num_inference_steps: DDIM denoising steps.
        cfg_scale: Classifier-free guidance scale.
        class_id: Specific class ID to generate, or -1 for grid across classes.
        attribute_names: Comma-separated CelebA attribute names for attribute conditioning.
        seed: Random seed for initial noise.
        output_path: File path to save the generated image grid.
        use_ema: Whether to restore EMA weights if present in checkpoint.

    Returns:
        Generated images array in NHWC format within [0.0, 1.0].
    """
    if isinstance(config, (str, Path)):
        config = load_config(str(config))

    metadata = get_dataset_metadata(config)
    rng = RNGManager(seed)

    model = create_model(config, rng.next())

    # Load checkpoint if provided
    if checkpoint:
        checkpoint_root, step = resolve_resume_checkpoint(checkpoint)
        checkpointer = CheckpointManager(str(checkpoint_root))
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        template = {
            "model": restore_template(nnx.state(model), sharding),
            "ema": restore_template(nnx.state(model), sharding),
        }
        ckpt_state = checkpointer.restore(step, args=restore_args(template, sharding))
        if ckpt_state is None:
            raise ValueError(f"Unable to load checkpoint at {checkpoint}")

        state_group = "ema" if (use_ema and "ema" in ckpt_state) else "model"
        target_group = ckpt_state[state_group]
        # In case EMA was stored under 'params' or direct state
        if "params" in target_group and "params" not in nnx.state(model).to_pure_dict():
            target_group = target_group["params"]

        validate_nnx_state(nnx.state(model), target_group, state_group)
        restore_nnx_state(nnx.state(model), target_group, state_group)
        print(f"Loaded {state_group.upper()} weights from step {step} ({checkpoint_root})")
    else:
        print("Warning: No checkpoint provided, sampling from random model weights.")

    evaluator = Evaluator(config, metadata)

    # Resolve conditioning inputs
    if metadata.label_mode == "class":
        if class_id >= 0:
            sample_labels = jnp.full((num_samples,), class_id, dtype=jnp.int32)
        else:
            sample_labels = jnp.arange(num_samples, dtype=jnp.int32) % metadata.num_classes
        null_labels = jnp.full((num_samples,), metadata.num_classes, dtype=jnp.int32)
    elif metadata.label_mode == "attributes":
        sample_labels = jnp.zeros((num_samples, metadata.label_dim or 40), dtype=jnp.int32)
        requested_attrs = [name.strip() for name in attribute_names.split(",") if name.strip()]
        for attr_name in requested_attrs:
            if attr_name not in CELEBA_ATTRIBUTE_NAMES:
                raise ValueError(
                    f"Unknown CelebA attribute: {attr_name!r}. "
                    f"Valid names include: {', '.join(CELEBA_ATTRIBUTE_NAMES[:5])}..."
                )
            attr_idx = CELEBA_ATTRIBUTE_NAMES.index(attr_name)
            sample_labels = sample_labels.at[:, attr_idx].set(1)
        null_labels = jnp.zeros_like(sample_labels)
    else:
        sample_labels = None
        null_labels = None

    sample_shape = (
        num_samples,
        config.model.input_size,
        config.model.input_size,
        config.model.in_channels,
    )

    @nnx.jit
    def model_fn(x, t, y):
        out = model(x, t, y)
        if out.shape[-1] == x.shape[-1] * 2:
            return jnp.split(out, 2, axis=-1)[0]
        return out

    sampler = DDIMSampler()
    latents_or_pixels = sampler.sample(
        model_fn=model_fn,
        shape=sample_shape,
        rng_key=rng.next(),
        num_inference_steps=num_inference_steps,
        y=sample_labels,
        null_y=null_labels,
        cfg_scale=cfg_scale,
    )

    images = evaluator.decode_samples(latents_or_pixels)

    if output_path:
        save_sample_grid(images, output_path)
        print(f"Saved {num_samples} samples to {output_path}")

    return images
