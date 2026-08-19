"""Sample raw-pixel Fashion MNIST images from a DiT checkpoint."""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl import flags
from flax import nnx
from orbax.checkpoint import type_handlers
from PIL import Image

from src.models.dit.dit import DiT, resolve_conditioning_mode
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager, materialize_checkpoint
from src.utils.config import load_config
from src.utils.rng import RNGManager

FLAGS = flags.FLAGS
if "config" not in FLAGS:
    flags.DEFINE_string(
        "config", "configs/fashion_mnist.yaml", "Training config used to build the model."
    )
if "checkpoint" not in FLAGS:
    flags.DEFINE_string(
        "checkpoint",
        "",
        "Local Orbax checkpoint or GCS checkpoint, for example "
        "outputs/fashion_mnist/checkpoints/1000 or "
        "gs://bucket/models/run/checkpoints. If a run/checkpoints directory is "
        "provided, the highest numeric checkpoint is selected automatically.",
    )
if "num_samples" not in FLAGS:
    flags.DEFINE_integer("num_samples", 16, "Number of images to generate.")
if "num_inference_steps" not in FLAGS:
    flags.DEFINE_integer("num_inference_steps", 50, "Number of DDIM denoising steps.")
if "cfg_scale" not in FLAGS:
    flags.DEFINE_float("cfg_scale", -1.0, "Override config classifier-free guidance scale.")
if "class_id" not in FLAGS:
    flags.DEFINE_integer("class_id", -1, "Class ID 0-9, or -1 for all classes.")
if "seed" not in FLAGS:
    flags.DEFINE_integer("seed", 0, "Random seed for the initial noise.")
if "output_path" not in FLAGS:
    flags.DEFINE_string(
        "output_path", "./fashion_mnist_samples.png", "Path for the output image grid."
    )
if "cpu_only" not in FLAGS:
    flags.DEFINE_bool(
        "cpu_only",
        False,
        "Force CPU execution and restore TPU checkpoints onto the local CPU.",
    )



def _build_model(config, seed: int, label_mode: str, label_dropout_prob: float) -> DiT:
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
        label_mode=label_mode,
        label_dropout_prob=label_dropout_prob,
        compute_dtype=(jnp.bfloat16 if config.training.get("use_bf16", False) else None),
        learn_sigma=config.model.get("learn_sigma", False),
        rngs=nnx.Rngs(rng.next()),
    )


def _restore_ema(model: DiT, checkpoint: str) -> None:
    """Restore EMA parameters from an individual Orbax checkpoint directory."""
    checkpoint_path = materialize_checkpoint(checkpoint)
    print(f"Using checkpoint: {checkpoint_path}")
    checkpoint_root = checkpoint_path.parent
    checkpoint_step = int(checkpoint_path.name)
    manager = CheckpointManager(checkpoint_root)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    pure_state = nnx.state(model).to_pure_dict()

    def template(value):
        if isinstance(value, dict):
            return {key: template(child) for key, child in value.items()}
        return {"value": jax.device_put(value, sharding)}

    restore_template = {"ema": template(pure_state)}
    restore_args = jax.tree.map(
        lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding),
        restore_template,
    )
    state = manager.restore(
        checkpoint_step,
        args=ocp.args.PyTreeRestore(
            restore_template,
            restore_args=restore_args,
            partial_restore=True,
        ),
    )
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
    if FLAGS.cpu_only:
        jax.config.update("jax_platforms", "cpu")

    config = load_config(FLAGS.config)
    conditioning = config.model.get("conditioning", "class")
    label_mode = resolve_conditioning_mode(conditioning)
    if FLAGS.class_id < -1 or FLAGS.class_id >= config.model.num_classes:
        raise ValueError("--class_id must be -1 or a valid Fashion MNIST class ID (0-9)")
    if conditioning == "unconditional" and FLAGS.class_id != -1:
        raise ValueError("--class_id cannot be used when model.conditioning is 'unconditional'")

    use_bf16 = config.training.get("use_bf16", False) and jax.devices()[0].platform == "tpu"
    if not use_bf16:
        config.training.use_bf16 = False
    model = _build_model(
        config,
        FLAGS.seed,
        label_mode,
        0.1 if conditioning == "class" else 0.0,
    )
    _restore_ema(model, FLAGS.checkpoint)
    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(x, t, y):
        output = model(x, t, y)
        if output.shape[-1] == x.shape[-1] * 2:
            return jnp.split(output, 2, axis=-1)[0]
        return output

    if conditioning == "class" and FLAGS.class_id >= 0:
        labels = jnp.full((FLAGS.num_samples,), FLAGS.class_id, dtype=jnp.int32)
    else:
        labels = jnp.arange(FLAGS.num_samples, dtype=jnp.int32) % config.model.num_classes
    null_labels = (
        jnp.full_like(labels, config.model.num_classes)
        if conditioning == "class"
        else None
    )
    cfg_scale = (
        config.evaluation.cfg_scale if FLAGS.cfg_scale < 0 else FLAGS.cfg_scale
    ) if conditioning == "class" else 1.0
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
        cfg_scale=cfg_scale,
    )
    _save_grid(samples, FLAGS.output_path)
    mode = (
        f"class {FLAGS.class_id}"
        if conditioning == "class" and FLAGS.class_id >= 0
        else conditioning
    )
    print(f"Generated {FLAGS.num_samples} Fashion MNIST samples ({mode}) on {jax.devices()[0]}.")
    print(f"Saved image grid to {FLAGS.output_path}")
