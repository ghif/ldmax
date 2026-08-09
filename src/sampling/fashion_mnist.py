"""Sample raw-pixel Fashion MNIST images from a DiT checkpoint."""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from flax import nnx
from PIL import Image

from src.models.dit.dit import DiT
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.rng import RNGManager


FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/fashion_mnist.yaml", "Training config used to build the model.")
flags.DEFINE_string("checkpoint", "", "Checkpoint directory, for example outputs/fashion_mnist/checkpoints/1000.")
flags.DEFINE_integer("num_samples", 16, "Number of images to generate.")
flags.DEFINE_integer("num_inference_steps", 50, "Number of DDIM denoising steps.")
flags.DEFINE_float("cfg_scale", 1.0, "Classifier-free guidance scale.")
flags.DEFINE_integer("seed", 0, "Random seed for the initial noise.")
flags.DEFINE_string("output_path", "./fashion_mnist_samples.png", "Path for the output image grid.")


def _build_model(config, seed: int) -> DiT:
    """Build the same model architecture used by the training runner."""
    rng = RNGManager(seed)
    return DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_classes=config.model.num_classes,
        learn_sigma=config.model.get("learn_sigma", False),
        rngs=nnx.Rngs(rng.next()),
    )


def _restore_ema(model: DiT, checkpoint: str) -> None:
    """Restore EMA parameters from an individual Orbax checkpoint directory."""
    checkpoint_root = os.path.dirname(os.path.abspath(checkpoint))
    checkpoint_step = int(os.path.basename(os.path.normpath(checkpoint)))
    manager = CheckpointManager(checkpoint_root)
    state = manager.restore(checkpoint_step)
    if state is None or "ema" not in state:
        raise ValueError(f"Checkpoint does not contain EMA parameters: {checkpoint}")

    # Orbax restores an NNX State as ordinary dictionaries.  NNX Lists use
    # integer paths in memory, while serialized checkpoint paths are strings,
    # so update the existing Variables by their flat paths.
    checkpoint_state = state["ema"]
    flat_state = nnx.state(model).flat_state()
    for path, variable in zip(flat_state.paths, flat_state.leaves):
        value = checkpoint_state
        for key in path:
            if not isinstance(value, dict):
                raise ValueError(f"Malformed EMA state at path {path}")
            value = value[key] if key in value else value[str(key)]
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        variable.value = value


def _save_grid(samples: jax.Array, output_path: str) -> None:
    """Save normalized ``[N, H, W, 1]`` samples as a grayscale image grid."""
    pixels = np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0)
    pixels = (pixels[..., 0] * 255.0).round().astype(np.uint8)
    count, height, width = pixels.shape
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    grid = np.zeros((rows * height, columns * width), dtype=np.uint8)
    for index, image in enumerate(pixels):
        row, column = divmod(index, columns)
        grid[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    Image.fromarray(grid, mode="L").save(output_path)


def main(_):
    """Generate a grid of Fashion MNIST samples from a trained checkpoint."""
    if not FLAGS.checkpoint:
        raise ValueError("--checkpoint is required")
    if FLAGS.num_samples < 1:
        raise ValueError("--num_samples must be positive")

    config = load_config(FLAGS.config)
    model = _build_model(config, FLAGS.seed)
    _restore_ema(model, FLAGS.checkpoint)
    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(x, t, y):
        output = model(x, t, y)
        if output.shape[-1] == x.shape[-1] * 2:
            return jnp.split(output, 2, axis=-1)[0]
        return output

    labels = jnp.arange(FLAGS.num_samples, dtype=jnp.int32) % config.model.num_classes
    null_labels = jnp.full_like(labels, config.model.num_classes)
    samples = sampler.sample(
        model_fn,
        (
            FLAGS.num_samples,
            config.model.input_size,
            config.model.input_size,
            config.model.in_channels,
        ),
        jax.random.key(FLAGS.seed),
        num_inference_steps=FLAGS.num_inference_steps,
        y=labels,
        null_y=null_labels,
        cfg_scale=FLAGS.cfg_scale,
    )
    _save_grid(samples, FLAGS.output_path)
    print(f"Generated {FLAGS.num_samples} Fashion MNIST samples on {jax.devices()[0]}.")
    print(f"Saved image grid to {FLAGS.output_path}")
