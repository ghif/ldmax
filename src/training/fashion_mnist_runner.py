"""End-to-end raw-pixel DiT training on Fashion MNIST."""

import json
import os
import time
from math import ceil, sqrt
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import flags
from flax import nnx
from orbax.checkpoint import type_handlers
from PIL import Image

from src.data.fashion_mnist import get_fashion_mnist_dataset
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
    flags.DEFINE_string("config", "configs/fashion_mnist.yaml", "Path to the config file.")
if "output_dir" not in FLAGS:
    flags.DEFINE_string("output_dir", "", "Directory for logs and checkpoints.")
if "resume_from" not in FLAGS:
    flags.DEFINE_string(
        "resume_from",
        "",
        "Run directory or checkpoint directory from which to resume training.",
    )


def _save_sample_grid(samples: jax.Array, path: str) -> None:
    """Save an NHWC batch of [0, 1] samples as a PNG grid."""
    images = np.asarray(samples)
    images = (images.clip(0.0, 1.0) * 255.0).round().astype(np.uint8)
    rows = ceil(len(images) / ceil(sqrt(len(images))))
    columns = ceil(sqrt(len(images)))
    height, width, channels = images.shape[1:]
    mode = "L" if channels == 1 else "RGB"
    grid = Image.new(mode, (columns * width, rows * height))

    for index, image in enumerate(images):
        if channels == 1:
            image = image[..., 0]
        else:
            image = image[..., :3]
        grid.paste(
            Image.fromarray(image, mode=mode),
            ((index % columns) * width, (index // columns) * height),
        )

    grid.save(path)


def _checkpoint_state(state):
    """Convert floating-point checkpoint leaves to FP32."""
    return jax.tree.map(
        lambda value: value.astype(jnp.float32)
        if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.floating)
        else value,
        state,
    )


def _resolve_resume_checkpoint(resume_from: str) -> tuple[Path, int]:
    """Resolve a run/checkpoint path to an Orbax root and step."""
    path = Path(resume_from).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {path}")

    if (path / "checkpoints").is_dir():
        checkpoint_root = path / "checkpoints"
        manager = CheckpointManager(str(checkpoint_root))
        step = manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found under {checkpoint_root}")
        return checkpoint_root, int(step)

    if path.name == "checkpoints" and path.is_dir():
        checkpoint_root = path
        manager = CheckpointManager(str(checkpoint_root))
        step = manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found under {checkpoint_root}")
        return checkpoint_root, int(step)

    if path.name.isdigit() and (path / "default").is_dir():
        return path.parent, int(path.name)

    raise ValueError(
        "--resume_from must be a run directory containing checkpoints/ or "
        "an individual Orbax checkpoint directory"
    )


def _checkpoint_has_rng(checkpoint_root: Path, step: int) -> bool:
    """Check Orbax metadata for an RNG leaf without deserializing arrays."""
    metadata_path = checkpoint_root / str(step) / "default" / "_METADATA"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return any(
        key.startswith("('rng'")
        for key in metadata.get("tree_metadata", {})
    )


def _checkpoint_value(state: Mapping[str, Any], path: tuple[Any, ...]) -> Any:
    """Look up a serialized NNX leaf using integer or string path keys."""
    value: Any = state
    for key in path:
        if not isinstance(value, Mapping):
            raise KeyError(path)
        if key in value:
            value = value[key]
        elif str(key) in value:
            value = value[str(key)]
        else:
            raise KeyError(path)
    if isinstance(value, Mapping) and "value" in value:
        value = value["value"]
    return value


def _validate_nnx_state(
    target_state: nnx.State,
    checkpoint_state: Mapping[str, Any],
    name: str,
) -> None:
    """Validate serialized paths and shapes before mutating an NNX state."""
    for path, variable in zip(
        target_state.flat_state().paths,
        target_state.flat_state().leaves,
    ):
        try:
            value = _checkpoint_value(checkpoint_state, path)
        except KeyError as error:
            raise ValueError(f"Checkpoint {name} is missing state path {path}") from error
        if hasattr(value, "shape") and value.shape != variable.value.shape:
            raise ValueError(
                f"Checkpoint {name} shape mismatch at {path}: "
                f"checkpoint={value.shape}, config={variable.value.shape}"
            )


def _restore_nnx_state(
    target_state: nnx.State,
    checkpoint_state: Mapping[str, Any],
    name: str,
) -> None:
    """Restore serialized values into an existing NNX state."""
    for path, variable in zip(
        target_state.flat_state().paths,
        target_state.flat_state().leaves,
    ):
        value = _checkpoint_value(checkpoint_state, path)
        if hasattr(value, "dtype") and hasattr(variable.value, "dtype"):
            if jnp.issubdtype(variable.value.dtype, jnp.floating):
                value = value.astype(variable.value.dtype)
        variable.value = value


def _restore_template(state: nnx.State) -> Any:
    """Build a concrete Orbax template from an NNX state."""
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def wrap(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: wrap(child) for key, child in value.items()}
        return {"value": jax.device_put(value, sharding)}

    return wrap(state.to_pure_dict())


def _restore_args(template: Any, sharding: jax.sharding.Sharding) -> Any:
    """Create Orbax array restore arguments for a concrete template."""
    args = jax.tree.map(
        lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding),
        template,
    )
    return ocp.args.PyTreeRestore(template, restore_args=args, partial_restore=True)


def _restore_training_state(
    model: DiT,
    optimizer: nnx.Optimizer,
    ema: EMAManager,
    checkpoint_state: Mapping[str, Any],
    conditioning: str,
) -> None:
    """Validate and restore model, optimizer, and EMA state."""
    required = {"model", "ema", "opt"}
    missing = required.difference(checkpoint_state)
    if missing:
        raise ValueError(f"Checkpoint is missing required state groups: {sorted(missing)}")

    model_state = nnx.state(model)
    optimizer_state = nnx.state(optimizer)
    ema_state = ema.ema_state
    if conditioning == "class":
        class_embedding = checkpoint_state["model"].get("y_embedder", {}).get("embedding_table")
        if class_embedding is None:
            raise ValueError("Checkpoint conditioning does not match config: expected class labels")
    else:
        class_embedding = checkpoint_state["model"].get("y_embedder", {}).get("embedding_table")
        if class_embedding is not None:
            raise ValueError(
                "Checkpoint conditioning does not match config: "
                "expected unconditional model"
            )

    _validate_nnx_state(model_state, checkpoint_state["model"], "model")
    _validate_nnx_state(optimizer_state, checkpoint_state["opt"], "optimizer")
    _validate_nnx_state(ema_state, checkpoint_state["ema"], "EMA")
    _restore_nnx_state(model_state, checkpoint_state["model"], "model")
    _restore_nnx_state(optimizer_state, checkpoint_state["opt"], "optimizer")
    _restore_nnx_state(ema_state, checkpoint_state["ema"], "EMA")


def main(_):
    """Train a class-conditional raw-pixel diffusion model."""
    config = load_config(FLAGS.config)
    output_dir = FLAGS.output_dir or "./outputs/fashion_mnist"
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
    label_mode = resolve_conditioning_mode(conditioning)
    label_dropout_prob = 0.1 if conditioning == "class" else 0.0
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
    sample_artifact_dir = os.path.join(output_dir, "checkpoints", "samples")
    os.makedirs(sample_artifact_dir, exist_ok=True)

    compute_dtype = jnp.bfloat16 if use_bf16 else None
    model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_classes=config.model.num_classes,
        label_mode=label_mode,
        label_dropout_prob=label_dropout_prob,
        learn_sigma=config.model.get("learn_sigma", False),
        compute_dtype=compute_dtype,
        rngs=nnx.Rngs(rng.next()),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        ),
        wrt=nnx.Param,
    )
    ema = EMAManager(model, decay=config.training.ema_decay)
    sampling_model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_classes=config.model.num_classes,
        label_mode=label_mode,
        label_dropout_prob=label_dropout_prob,
        learn_sigma=config.model.get("learn_sigma", False),
        compute_dtype=compute_dtype,
        rngs=nnx.Rngs(rng.next()),
    )
    if resume_root is not None:
        source_checkpointer = CheckpointManager(str(resume_root))
        restore_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        restore_template = {
            "model": _restore_template(nnx.state(model)),
            "ema": _restore_template(ema.ema_state),
            "opt": _restore_template(nnx.state(optimizer)),
        }
        if checkpoint_has_rng:
            restore_template["rng"] = {"key": jax.device_put(rng.state, restore_sharding)}
        checkpoint_state = source_checkpointer.restore(
            restored_step,
            args=_restore_args(restore_template, restore_sharding),
        )
        if checkpoint_state is None:
            raise ValueError(f"Unable to restore checkpoint step {restored_step}")
        _restore_training_state(model, optimizer, ema, checkpoint_state, conditioning)
        rng_state = checkpoint_state.get("rng")
        if isinstance(rng_state, Mapping) and "key" in rng_state:
            rng.restore(rng_state["key"])
        else:
            rng = RNGManager.from_seed_and_step(config.training.seed, restored_step)
    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(model, x, t, y):
        output = model(x, t, y)
        if output.shape[-1] == x.shape[-1] * 2:
            return jnp.split(output, 2, axis=-1)[0]
        return output

    dataset = get_fashion_mnist_dataset(
        batch_size=config.training.batch_size,
        split="train",
        shuffle=True,
        seed=config.training.seed,
        dataset_name=config.data.dataset_name,
    )
    prefetch_size = config.training.get("prefetch_size", 2)
    if prefetch_size > 0:
        prefetch_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        dataset_iter = iter(DevicePrefetcher(dataset, prefetch_sharding, prefetch_size))
    else:
        dataset_iter = iter(dataset)
    validation_dataset = get_fashion_mnist_dataset(
        batch_size=config.training.batch_size,
        split="test",
        shuffle=False,
        seed=config.training.seed,
        dataset_name=config.data.dataset_name,
    )
    if prefetch_size > 0:
        validation_iter = iter(DevicePrefetcher(validation_dataset, prefetch_sharding, 1))
    else:
        validation_iter = iter(validation_dataset)

    @nnx.jit
    def validation_step(model, latents, labels, rng_key):
        return compute_loss(model, latents, labels, rng_key, train=False)

    trainlog_path = os.path.join(output_dir, "train_logs.txt")
    trainlog = open(trainlog_path, "w", encoding="utf-8")

    def emit(message: str) -> None:
        """Write the same human-readable message to the terminal and log file."""
        print(message, flush=True)
        trainlog.write(message + "\n")
        trainlog.flush()

    emit(f"Using device: {jax.devices()[0]}")
    emit("Training directly on 28x28x1 pixels; no VAE is used.")
    if resume_root is not None:
        emit(f"Resuming from {resume_root} at checkpoint step {restored_step}.")
        if checkpoint_state is not None and not checkpoint_has_rng:
            emit("Checkpoint has no RNG state; using deterministic seed-and-step fallback.")
    emit(f"Starting training for {config.training.total_steps} steps...")
    if resume_root is not None and restored_step >= config.training.total_steps:
        emit("Checkpoint already reaches the configured target; no training steps to run.")
    training_start = time.perf_counter()
    latest_validation_loss = None
    data_wait_duration = 0.0
    ema_update_duration = 0.0
    sampling_duration = 0.0
    gcs_sync_duration = 0.0

    try:
        for step in range(restored_step, config.training.total_steps):
            data_wait_start = time.perf_counter()
            batch = next(dataset_iter)
            data_wait_duration = time.perf_counter() - data_wait_start
            train_step_start = time.perf_counter()
            metrics = train_step(
                model,
                optimizer,
                batch["image"],
                batch["label"],
                rng.next(),
                use_bf16=use_bf16,
            )
            train_dispatch_duration = time.perf_counter() - train_step_start
            ema_start = time.perf_counter()
            ema.update(model)
            ema_update_duration = time.perf_counter() - ema_start

            if step % config.evaluation.log_interval == 0:
                loss = float(metrics["loss"])
                train_step_duration = time.perf_counter() - train_step_start
                validation_batch = next(validation_iter)
                validation_step_start = time.perf_counter()
                validation_loss = float(
                    validation_step(
                        model,
                        validation_batch["image"],
                        validation_batch["label"],
                        rng.next(),
                    )
                )
                latest_validation_loss = validation_loss
                validation_step_duration = time.perf_counter() - validation_step_start
                batch_size = batch["image"].shape[0]
                validation_batch_size = validation_batch["image"].shape[0]
                train_steps_per_sec = 1.0 / max(train_step_duration, 1e-12)
                validation_steps_per_sec = 1.0 / max(validation_step_duration, 1e-12)
                train_samples_per_sec = batch_size * train_steps_per_sec
                validation_samples_per_sec = validation_batch_size * validation_steps_per_sec
                elapsed_sec = time.perf_counter() - training_start
                progress_percent = 100.0 * (step + 1) / config.training.total_steps
                logger.log_scalars(
                    step,
                    {
                        "train/loss": loss,
                        "validation/loss": validation_loss,
                        "performance/train_step_sec": train_step_duration,
                        "performance/train_steps_per_sec": train_steps_per_sec,
                        "performance/train_samples_per_sec": train_samples_per_sec,
                        "performance/data_wait_sec": data_wait_duration,
                        "performance/train_dispatch_sec": train_dispatch_duration,
                        "performance/ema_update_sec": ema_update_duration,
                        "performance/validation_step_sec": validation_step_duration,
                        "performance/validation_steps_per_sec": validation_steps_per_sec,
                        "performance/validation_samples_per_sec": validation_samples_per_sec,
                        "performance/sampling_sec": sampling_duration,
                        "performance/gcs_sync_sec": gcs_sync_duration,
                    },
                )
                logger.flush()
                emit(
                    f"step={step + 1:05d}/{config.training.total_steps:05d} "
                    f"progress={progress_percent:.2f}% train_loss={loss:.6f} "
                    f"validation_loss={validation_loss:.6f} "
                    f"train_step_sec={train_step_duration:.6f} "
                    f"train_steps_per_sec={train_steps_per_sec:.6f} "
                    f"train_samples_per_sec={train_samples_per_sec:.6f} "
                    f"data_wait_sec={data_wait_duration:.6f} "
                    f"train_dispatch_sec={train_dispatch_duration:.6f} "
                    f"ema_update_sec={ema_update_duration:.6f} "
                    f"validation_step_sec={validation_step_duration:.6f} "
                    f"validation_steps_per_sec={validation_steps_per_sec:.6f} "
                    f"validation_samples_per_sec={validation_samples_per_sec:.6f} "
                    f"sampling_sec={sampling_duration:.6f} "
                    f"gcs_sync_sec={gcs_sync_duration:.6f} "
                    f"elapsed_sec={elapsed_sec:.6f}"
                )

            if step % config.evaluation.sampling_interval == 0:
                sampling_start = time.perf_counter()
                nnx.update(sampling_model, ema.ema_state)
                sample_count = config.evaluation.sample_count
                labels = jnp.arange(sample_count, dtype=jnp.int32) % config.model.num_classes
                null_labels = (
                    jnp.full_like(labels, config.model.num_classes)
                    if conditioning == "class"
                    else None
                )
                samples = sampler.sample(
                    lambda x, t, y: model_fn(sampling_model, x, t, y),
                    (
                        sample_count,
                        config.model.input_size,
                        config.model.input_size,
                        config.model.in_channels,
                    ),
                    rng.next(),
                    num_inference_steps=config.evaluation.num_inference_steps,
                    y=labels,
                    null_y=null_labels,
                    cfg_scale=(
                        config.evaluation.cfg_scale
                        if conditioning == "class"
                        else 1.0
                    ),
                )
                sample_images = (samples + 1.0).clip(0.0, 2.0) / 2.0
                logger.log_images(step, "train/samples", sample_images)
                _save_sample_grid(
                    sample_images,
                    os.path.join(sample_artifact_dir, f"samples_step_{step:06d}.png"),
                )
                logger.flush()
                sampling_duration = time.perf_counter() - sampling_start

            should_checkpoint = (
                step % config.evaluation.checkpoint_interval == 0 and step > 0
            ) or step == config.training.total_steps - 1
            if should_checkpoint:
                if step % config.evaluation.log_interval != 0:
                    validation_batch = next(validation_iter)
                    latest_validation_loss = float(
                        validation_step(
                            model,
                            validation_batch["image"],
                            validation_batch["label"],
                            rng.next(),
                        )
                    )
                gcs_sync_start = time.perf_counter()
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
                gcs_sync_duration = time.perf_counter() - gcs_sync_start
        emit("Training complete.")
    finally:
        logger.close()
        trainlog.close()
        checkpointer.sync_to_gcs()
