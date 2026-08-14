"""Generate native-pixel CIFAR10 samples from a DiT checkpoint."""

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from absl import app, flags
from flax import nnx
from orbax.checkpoint import type_handlers
from PIL import Image

from src.models.dit.dit import DiT, resolve_conditioning_mode
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager, materialize_checkpoint
from src.utils.config import load_config
from src.utils.rng import RNGManager

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", "configs/cifar10_pixel.yaml", "Training config used to build the model."
)
flags.DEFINE_string("checkpoint", "", "Local Orbax checkpoint or checkpoint directory.")
flags.DEFINE_integer("num_samples", 16, "Number of images to generate.")
flags.DEFINE_integer("num_inference_steps", 50, "Number of DDIM denoising steps.")
flags.DEFINE_float("cfg_scale", -1.0, "Override configured classifier-free guidance scale.")
flags.DEFINE_integer("class_id", -1, "Class ID 0-9, or -1 for a class sweep.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("output_path", "./cifar10_samples.png", "Output image grid path.")
flags.DEFINE_bool("cpu_only", False, "Force CPU execution.")


def _restore_ema(model: DiT, checkpoint: str) -> None:
    """Restore EMA weights from an individual or latest checkpoint."""
    path = materialize_checkpoint(checkpoint)
    manager = CheckpointManager(path.parent)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def template(value):
        if isinstance(value, dict):
            return {key: template(child) for key, child in value.items()}
        return {"value": jax.device_put(value, sharding)}

    restore_template = {"ema": template(nnx.state(model).to_pure_dict())}
    restore_args = jax.tree.map(
        lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding), restore_template
    )
    state = manager.restore(
        int(path.name),
        args=ocp.args.PyTreeRestore(
            restore_template, restore_args=restore_args, partial_restore=True
        ),
    )
    if state is None or "ema" not in state:
        raise ValueError(f"Checkpoint does not contain EMA parameters: {checkpoint}")
    source = state["ema"]
    for path_tuple, variable in zip(
        nnx.state(model).flat_state().paths, nnx.state(model).flat_state().leaves
    ):
        value = source
        for key in path_tuple:
            value = value[key] if key in value else value[str(key)]
        variable.value = value["value"] if isinstance(value, dict) and "value" in value else value


def _save_grid(samples: jax.Array, output_path: str) -> None:
    """Save normalized RGB samples as a grid."""
    pixels = (np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0) * 255).round().astype(np.uint8)
    columns = math.ceil(math.sqrt(len(pixels)))
    rows = math.ceil(len(pixels) / columns)
    height, width = pixels.shape[1:3]
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(pixels):
        grid.paste(
            Image.fromarray(image[..., :3], mode="RGB"),
            ((index % columns) * width, (index // columns) * height),
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    grid.save(output_path)


def main(_):
    """Generate CIFAR10 samples."""
    if not FLAGS.checkpoint:
        raise ValueError("--checkpoint is required")
    if FLAGS.num_samples < 1:
        raise ValueError("--num_samples must be positive")
    if FLAGS.cpu_only:
        jax.config.update("jax_platforms", "cpu")
    config = load_config(FLAGS.config)
    conditioning = config.model.get("conditioning", "class")
    if conditioning == "class" and not -1 <= FLAGS.class_id < config.model.num_classes:
        raise ValueError("--class_id must be -1 or a valid CIFAR10 class ID (0-9)")
    if conditioning == "unconditional" and FLAGS.class_id != -1:
        raise ValueError("--class_id cannot be used with an unconditional model")
    model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_classes=config.model.num_classes,
        label_mode=resolve_conditioning_mode(conditioning),
        label_dropout_prob=0.1 if conditioning == "class" else 0.0,
        learn_sigma=config.model.get("learn_sigma", False),
        compute_dtype=jnp.bfloat16 if config.training.get("use_bf16", False) else None,
        rngs=nnx.Rngs(RNGManager(FLAGS.seed).next()),
    )
    _restore_ema(model, FLAGS.checkpoint)
    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(x, t, y):
        output = model(x, t, y)
        return jnp.split(output, 2, axis=-1)[0] if output.shape[-1] == x.shape[-1] * 2 else output

    labels = (
        jnp.full((FLAGS.num_samples,), FLAGS.class_id, dtype=jnp.int32)
        if conditioning == "class" and FLAGS.class_id >= 0
        else jnp.arange(FLAGS.num_samples, dtype=jnp.int32) % config.model.num_classes
    )
    null_labels = (
        jnp.full_like(labels, config.model.num_classes) if conditioning == "class" else None
    )
    cfg_scale = config.evaluation.get("cfg_scale", 1.5) if FLAGS.cfg_scale < 0 else FLAGS.cfg_scale
    samples = sampler.sample(
        model_fn,
        (FLAGS.num_samples, 32, 32, 3),
        jax.random.key(FLAGS.seed),
        num_inference_steps=FLAGS.num_inference_steps,
        y=labels,
        null_y=null_labels,
        cfg_scale=cfg_scale if conditioning == "class" else 1.0,
        clip_denoised=True,
    )
    _save_grid(samples, FLAGS.output_path)
    print(f"Generated {FLAGS.num_samples} CIFAR10 samples ({conditioning}) on {jax.devices()[0]}.")
    print(f"Saved image grid to {FLAGS.output_path}")


if __name__ == "__main__":
    app.run(main)
