"""Standalone inference script for DiT."""

import os
from absl import app, flags
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.models.dit.dit import DiT
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager
from src.utils.vae import VAEManager
from src.utils.rng import RNGManager

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "", "Path to the checkpoint.")
flags.DEFINE_integer("num_samples", 16, "Number of images to generate.")
flags.DEFINE_integer("num_steps", 50, "Number of sampling steps.")
flags.DEFINE_float("cfg_scale", 1.5, "Classifier-Free Guidance scale.")
flags.DEFINE_string("output_path", "./samples.png", "Output file path.")

def main(_):
    # 1. Setup RNG
    rng_manager = RNGManager(42)
    
    # 2. Initialize Model (Architecture must match training)
    # For MVP, we use hardcoded small dims matching cifar10.yaml baseline
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=128,
        depth=4,
        num_heads=4,
        num_classes=10,
        rngs=nnx.Rngs(rng_manager.next())
    )
    
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
    
    @nnx.jit
    def model_fn(x, t, y):
        out = model(x, t, y)
        if out.shape[-1] == x.shape[-1] * 2:
            return jnp.split(out, 2, axis=-1)[0]
        return out
        
    sample_labels = jnp.zeros((FLAGS.num_samples,), dtype=jnp.int32) # default class 0
    sample_shape = (FLAGS.num_samples, 32, 32, 3)
    
    print(f"Sampling {FLAGS.num_samples} images...")
    samples = sampler.sample(
        model_fn, 
        sample_shape, 
        rng_manager.next(), 
        num_inference_steps=FLAGS.num_steps,
        y=sample_labels,
        cfg_scale=FLAGS.cfg_scale
    )
    
    # 5. Save Grid
    grid_size = int(FLAGS.num_samples ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < FLAGS.num_samples:
            img = (np.array(samples[i]) * 255).astype(np.uint8) if hasattr(samples, "device") else samples[i]
            # Simple [0, 1] scaling for visualization
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            ax.imshow(img)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(FLAGS.output_path)
    print(f"Saved samples to {FLAGS.output_path}")

if __name__ == "__main__":
    app.run(main)
