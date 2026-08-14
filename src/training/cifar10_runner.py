"""End-to-end native-pixel diffusion training on CIFAR10."""

import json
import os
from collections.abc import Mapping
from math import ceil, sqrt
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import flags
from flax import nnx
from orbax.checkpoint import type_handlers
from PIL import Image

from src.data.cifar import get_cifar10_dataset
from src.models.dit.dit import DiT, resolve_conditioning_mode
from src.training.ema import EMAManager
from src.training.sampler import DDIMSampler
from src.training.step import compute_loss, train_step
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.logging import TensorBoardLogger
from src.utils.prefetch import DevicePrefetcher
from src.utils.rng import RNGManager

FLAGS = flags.FLAGS
if "config" not in FLAGS:
    flags.DEFINE_string("config", "configs/cifar10_pixel.yaml", "Path to the config file.")
if "output_dir" not in FLAGS:
    flags.DEFINE_string("output_dir", "", "Directory for logs and checkpoints.")
if "resume_from" not in FLAGS:
    flags.DEFINE_string("resume_from", "", "Run or checkpoint directory to resume from.")


def _checkpoint_state(state):
    """Convert floating-point checkpoint leaves to FP32."""
    return jax.tree.map(
        lambda value: value.astype(jnp.float32)
        if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.floating)
        else value,
        state,
    )


def _checkpoint_has_rng(checkpoint_root, step: int) -> bool:
    """Check Orbax metadata for an RNG leaf."""
    metadata_path = checkpoint_root / str(step) / "default" / "_METADATA"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return any(key.startswith("('rng'") for key in metadata.get("tree_metadata", {}))


def _resolve_resume_checkpoint(resume_from: str):
    """Resolve a run/checkpoint path to an Orbax root and step."""
    path = Path(resume_from).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {path}")
    if (path / "checkpoints").is_dir():
        root = path / "checkpoints"
        step = CheckpointManager(str(root)).latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found under {root}")
        return root, int(step)
    if path.name == "checkpoints" and path.is_dir():
        step = CheckpointManager(str(path)).latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found under {path}")
        return path, int(step)
    if path.name.isdigit() and (path / "default").is_dir():
        return path.parent, int(path.name)
    raise ValueError("--resume_from must be a run directory or Orbax checkpoint directory")


def _restore_template(state):
    """Build a concrete single-device Orbax restore template."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def wrap(value):
        if isinstance(value, Mapping):
            return {key: wrap(child) for key, child in value.items()}
        return {"value": jax.device_put(value, sharding)}

    return wrap(state.to_pure_dict())


def _restore_args(template, sharding):
    """Build array restore arguments for an NNX state template."""
    args = jax.tree.map(lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding), template)
    return ocp.args.PyTreeRestore(template, restore_args=args, partial_restore=True)


def _checkpoint_value(state, path):
    value = state
    for key in path:
        if not isinstance(value, Mapping):
            raise KeyError(path)
        if key in value:
            value = value[key]
        elif str(key) in value:
            value = value[str(key)]
        else:
            raise KeyError(path)
    return value["value"] if isinstance(value, Mapping) and "value" in value else value


def _restore_nnx_state(target_state, checkpoint_state, name: str) -> None:
    for path, variable in zip(target_state.flat_state().paths, target_state.flat_state().leaves):
        try:
            value = _checkpoint_value(checkpoint_state, path)
        except KeyError as error:
            raise ValueError(f"Checkpoint {name} is missing state path {path}") from error
        if hasattr(value, "shape") and value.shape != variable.value.shape:
            raise ValueError(f"Checkpoint {name} shape mismatch at {path}")
        if hasattr(value, "dtype") and jnp.issubdtype(variable.value.dtype, jnp.floating):
            value = value.astype(variable.value.dtype)
        variable.value = value


def _restore_training_state(model, optimizer, ema, checkpoint_state, conditioning):
    """Restore model, optimizer, and EMA state after shape validation."""
    required = {"model", "ema", "opt"}
    missing = required.difference(checkpoint_state)
    if missing:
        raise ValueError(f"Checkpoint is missing required state groups: {sorted(missing)}")
    class_embedding = checkpoint_state["model"].get("y_embedder", {}).get("embedding_table")
    if conditioning == "class" and class_embedding is None:
        raise ValueError("Checkpoint conditioning does not match config: expected class labels")
    if conditioning == "unconditional" and class_embedding is not None:
        raise ValueError(
            "Checkpoint conditioning does not match config: expected unconditional model"
        )
    _restore_nnx_state(nnx.state(model), checkpoint_state["model"], "model")
    _restore_nnx_state(nnx.state(optimizer), checkpoint_state["opt"], "optimizer")
    _restore_nnx_state(ema.ema_state, checkpoint_state["ema"], "EMA")


def _save_sample_grid(samples: jax.Array, path: str) -> None:
    """Save an NHWC batch in [0, 1] as an RGB grid."""
    images = (np.asarray(samples).clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    columns = ceil(sqrt(len(images)))
    rows = ceil(len(images) / columns)
    height, width = images.shape[1:3]
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        grid.paste(
            Image.fromarray(image[..., :3], mode="RGB"),
            ((index % columns) * width, (index // columns) * height),
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    grid.save(path)


def _build_model(config, rng_key: jax.Array) -> DiT:
    """Build the native-pixel CIFAR10 DiT from configuration."""
    conditioning = config.model.get("conditioning", "class")
    return DiT(
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
        rngs=nnx.Rngs(rng_key),
    )


def _validate_config(config) -> None:
    """Validate shape and conditioning invariants before model construction."""
    if config.model.input_size != 32 or config.model.in_channels != 3:
        raise ValueError("CIFAR10 pixel training requires input_size=32 and in_channels=3")
    if config.model.input_size % config.model.patch_size != 0:
        raise ValueError("model.input_size must be divisible by model.patch_size")
    if config.model.num_classes != 10:
        raise ValueError("CIFAR10 requires model.num_classes=10")
    if config.model.get("conditioning", "class") not in {"class", "unconditional"}:
        raise ValueError("model.conditioning must be 'class' or 'unconditional'")


def main(_):
    """Train a class-conditional or unconditional raw-pixel CIFAR10 model."""
    config = load_config(FLAGS.config)
    _validate_config(config)
    output_dir = FLAGS.output_dir or "./outputs/cifar10_pixel"
    if FLAGS.resume_from and not FLAGS.output_dir:
        raise ValueError("--output_dir is required when --resume_from is used")
    os.makedirs(output_dir, exist_ok=True)

    resume_root = None
    restored_step = 0
    checkpoint_state = None
    checkpoint_has_rng = False
    if FLAGS.resume_from:
        resume_root, restored_step = _resolve_resume_checkpoint(FLAGS.resume_from)
        checkpoint_has_rng = _checkpoint_has_rng(resume_root, restored_step)

    conditioning = config.model.get("conditioning", "class")
    use_bf16 = config.training.get("use_bf16", False)
    if use_bf16:
        jax.config.update("jax_default_matmul_precision", "bfloat16")
    rng = RNGManager(config.training.seed)
    logger = TensorBoardLogger(os.path.join(output_dir, "logs"))
    checkpointer = CheckpointManager(
        os.path.join(output_dir, "checkpoints"),
        gcs_directory="gs://diffjax/models",
        best_metric="validation_loss",
        best_mode="min",
        artifact_paths=[
            os.path.join(output_dir, "logs"),
            os.path.join(output_dir, "train_logs.txt"),
        ],
    )
    sample_dir = os.path.join(output_dir, "checkpoints", "samples")
    os.makedirs(sample_dir, exist_ok=True)

    model = _build_model(config, rng.next())
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(config.training.learning_rate, weight_decay=config.training.weight_decay),
        wrt=nnx.Param,
    )
    ema = EMAManager(model, decay=config.training.ema_decay)
    sampling_model = _build_model(config, rng.next())

    if resume_root is not None:
        source_checkpointer = CheckpointManager(str(resume_root))
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        template = {
            "model": _restore_template(nnx.state(model)),
            "ema": _restore_template(ema.ema_state),
            "opt": _restore_template(nnx.state(optimizer)),
        }
        if checkpoint_has_rng:
            template["rng"] = {"key": jax.device_put(rng.state, sharding)}
        checkpoint_state = source_checkpointer.restore(
            restored_step, args=_restore_args(template, sharding)
        )
        if checkpoint_state is None:
            raise ValueError(f"Unable to restore checkpoint step {restored_step}")
        _restore_training_state(model, optimizer, ema, checkpoint_state, conditioning)
        if isinstance(checkpoint_state.get("rng"), dict):
            rng.restore(checkpoint_state["rng"]["key"])
        else:
            rng = RNGManager.from_seed_and_step(config.training.seed, restored_step)

    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(model, x, t, y):
        output = model(x, t, y)
        return jnp.split(output, 2, axis=-1)[0] if output.shape[-1] == x.shape[-1] * 2 else output

    data = get_cifar10_dataset(
        batch_size=config.training.batch_size,
        split="train",
        shuffle=True,
        seed=config.training.seed,
    )
    validation_data = get_cifar10_dataset(
        batch_size=config.training.batch_size,
        split="test",
        shuffle=False,
        seed=config.training.seed,
    )
    prefetch_size = config.training.get("prefetch_size", 2)
    if prefetch_size > 0:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        data_iter = iter(DevicePrefetcher(data, sharding, prefetch_size))
        validation_iter = iter(DevicePrefetcher(validation_data, sharding, 1))
    else:
        data_iter, validation_iter = iter(data), iter(validation_data)

    @nnx.jit
    def validation_step(model, images, labels, key):
        return compute_loss(model, images, labels, key, train=False)

    log_path = os.path.join(output_dir, "train_logs.txt")
    trainlog = open(log_path, "w", encoding="utf-8")

    def emit(message: str) -> None:
        print(message, flush=True)
        trainlog.write(message + "\n")
        trainlog.flush()

    emit(f"Using device: {jax.devices()[0]}")
    emit(f"Training CIFAR10 directly on 32x32x3 pixels; conditioning={conditioning}.")
    if resume_root is not None:
        emit(f"Resuming from {resume_root} at checkpoint step {restored_step}.")
    latest_validation_loss = None
    try:
        for step in range(restored_step, config.training.total_steps):
            batch = next(data_iter)
            metrics = train_step(
                model, optimizer, batch["image"], batch["label"], rng.next(), use_bf16=use_bf16
            )
            ema.update(model)

            if step % config.evaluation.log_interval == 0:
                validation_batch = next(validation_iter)
                latest_validation_loss = float(
                    validation_step(
                        model, validation_batch["image"], validation_batch["label"], rng.next()
                    )
                )
                logger.log_scalars(
                    step,
                    {
                        "train/loss": float(metrics["loss"]),
                        "validation/loss": latest_validation_loss,
                    },
                )
                logger.flush()
                emit(
                    f"step={step + 1:05d}/{config.training.total_steps:05d} "
                    f"train_loss={float(metrics['loss']):.6f} "
                    f"validation_loss={latest_validation_loss:.6f}"
                )

            if step % config.evaluation.sampling_interval == 0:
                nnx.update(sampling_model, ema.ema_state)
                count = config.evaluation.get("sample_count", 16)
                labels = jnp.arange(count, dtype=jnp.int32) % config.model.num_classes
                null_labels = (
                    jnp.full_like(labels, config.model.num_classes)
                    if conditioning == "class"
                    else None
                )
                samples = sampler.sample(
                    lambda x, t, y: model_fn(sampling_model, x, t, y),
                    (count, 32, 32, 3),
                    rng.next(),
                    num_inference_steps=config.evaluation.get("num_inference_steps", 50),
                    y=labels,
                    null_y=null_labels,
                    cfg_scale=config.evaluation.get("cfg_scale", 1.5)
                    if conditioning == "class"
                    else 1.0,
                    clip_denoised=True,
                )
                images = (samples + 1.0).clip(0.0, 2.0) / 2.0
                logger.log_images(step, "train/samples", images)
                _save_sample_grid(images, os.path.join(sample_dir, f"samples_step_{step:06d}.png"))
                logger.flush()

            if (
                step % config.evaluation.checkpoint_interval == 0 and step > 0
            ) or step == config.training.total_steps - 1:
                if latest_validation_loss is None:
                    validation_batch = next(validation_iter)
                    latest_validation_loss = float(
                        validation_step(
                            model, validation_batch["image"], validation_batch["label"], rng.next()
                        )
                    )
                checkpointer.save(
                    step,
                    {
                        "model": _checkpoint_state(nnx.state(model)),
                        "ema": _checkpoint_state(ema.ema_state),
                        "opt": _checkpoint_state(nnx.state(optimizer)),
                        "rng": {"key": rng.state},
                    },
                    metrics={"validation_loss": latest_validation_loss},
                )
        emit("Training complete.")
    finally:
        logger.close()
        trainlog.close()
        checkpointer.sync_to_gcs()
