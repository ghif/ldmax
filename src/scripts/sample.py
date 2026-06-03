"""Standalone inference script for DiT."""

import os
import math
from absl import app, flags
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from src.models.dit.dit import DiT
from src.models.unet.unet import UNetModel
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager
from src.utils.vae import VAEManager
from src.utils.rng import RNGManager
from src.utils.config import load_config
from src.data.celeba import CELEBA_ATTRIBUTE_NAMES

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/cifar10.yaml", "Path to the config file.")
flags.DEFINE_string("checkpoint", "", "Path to the checkpoint.")
flags.DEFINE_integer("num_samples", 16, "Number of images to generate.")
flags.DEFINE_integer("num_steps", 50, "Number of sampling steps.")
flags.DEFINE_float("cfg_scale", 1.5, "Classifier-Free Guidance scale.")
flags.DEFINE_string("output_path", "./samples.png", "Output file path.")
flags.DEFINE_integer("class_id", 0, "CIFAR-10 class id to sample when using a class-conditioned model.")
flags.DEFINE_string(
    "attribute_names",
    "Smiling",
    "Comma-separated CelebA attribute names to activate when sampling an attribute-conditioned model.",
)

def main(_):
    # 1. Setup RNG
    rng_manager = RNGManager(42)

    # 2. Load config and build the matching model
    config = load_config(FLAGS.config)
    use_vae = config.model.get("use_vae", True)
    label_mode = getattr(config.model, "label_mode", "class")
    label_dim = getattr(config.model, "label_dim", None)
    if label_mode == "attribute" and label_dim is None:
        from src.data.celeba import CELEBA_ATTRIBUTE_NAMES
        label_dim = len(CELEBA_ATTRIBUTE_NAMES)

    # Initialize Model (Architecture must match training)
    model_type = config.model.get("type", "dit")
    if model_type == "dit":
        model = DiT(
            input_size=config.model.input_size,
            patch_size=config.model.patch_size,
            in_channels=config.model.in_channels,
            hidden_size=config.model.hidden_size,
            depth=config.model.depth,
            num_heads=config.model.num_heads,
            num_classes=config.model.num_classes,
            label_mode=label_mode,
            label_dim=label_dim,
            learn_sigma=config.model.get("learn_sigma", False),
            rngs=nnx.Rngs(rng_manager.next())
        )
    elif model_type == "unet":
        model = UNetModel(
            in_channels=config.model.in_channels,
            out_channels=config.model.get("out_channels", config.model.in_channels),
            model_channels=config.model.model_channels,
            attention_resolutions=config.model.attention_resolutions,
            num_res_blocks=config.model.num_res_blocks,
            channel_mult=config.model.channel_mult,
            num_heads=config.model.num_heads,
            transformer_depth=config.model.get("transformer_depth", 1),
            context_dim=config.model.get("context_dim", None),
            num_classes=config.model.num_classes,
            label_mode=label_mode,
            label_dim=label_dim,
            rngs=nnx.Rngs(rng_manager.next())
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 3. Load Checkpoint
    if FLAGS.checkpoint:
        checkpointer = CheckpointManager(os.path.dirname(FLAGS.checkpoint))
        step = int(os.path.basename(FLAGS.checkpoint))
        state = checkpointer.restore(step)
        nnx.update(model, state["model"])
        print(f"Restored from step {step}")
    else:
        print("Warning: No checkpoint provided, sampling from random initialization.")

    # 4. Sample
    sampler = DDIMSampler()
    vae_manager = VAEManager() if use_vae else None
    
    @nnx.jit
    def model_fn(x, t, y):
        out = model(x, t, y)
        if out.shape[-1] == x.shape[-1] * 2:
            return jnp.split(out, 2, axis=-1)[0]
        return out

    if config.dataset == "cifar10":
        sample_labels = jnp.full((FLAGS.num_samples,), FLAGS.class_id, dtype=jnp.int32)
        null_labels = jnp.full((FLAGS.num_samples,), config.model.num_classes, dtype=jnp.int32)
    elif config.dataset == "celeba":
        sample_labels = jnp.zeros((FLAGS.num_samples, config.model.label_dim), dtype=jnp.int32)
        requested_attributes = [name.strip() for name in FLAGS.attribute_names.split(",") if name.strip()]
        for attr_name in requested_attributes:
            if attr_name not in CELEBA_ATTRIBUTE_NAMES:
                raise ValueError(
                    f"Unknown CelebA attribute: {attr_name}. "
                    f"Valid names include: {', '.join(CELEBA_ATTRIBUTE_NAMES[:5])}, ..."
                )
            attr_idx = CELEBA_ATTRIBUTE_NAMES.index(attr_name)
            sample_labels = sample_labels.at[:, attr_idx].set(1)
        null_labels = jnp.zeros_like(sample_labels)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    sample_shape = (
        FLAGS.num_samples,
        config.model.input_size,
        config.model.input_size,
        config.model.in_channels,
    )
    
    print(f"Sampling {FLAGS.num_samples} images...")
    samples = sampler.sample(
        model_fn, 
        sample_shape, 
        rng_manager.next(), 
        num_inference_steps=FLAGS.num_steps,
        y=sample_labels,
        null_y=null_labels,
        cfg_scale=FLAGS.cfg_scale
    )

    # Decode from latent space back to pixel space for visualization.
    if use_vae:
        samples = vae_manager.decode(samples)
    else:
        samples = (samples / 2 + 0.5).clip(0, 1)
    
    # 5. Save Grid
    grid_size = int(math.ceil(FLAGS.num_samples ** 0.5))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < FLAGS.num_samples:
            img = np.asarray(samples[i])
            ax.imshow(img)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(FLAGS.output_path)
    print(f"Saved samples to {FLAGS.output_path}")

if __name__ == "__main__":
    app.run(main)
