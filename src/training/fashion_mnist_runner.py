"""End-to-end raw-pixel DiT training on Fashion MNIST."""

import os
import time
from math import ceil, sqrt

import jax
import jax.numpy as jnp
import numpy as np
from absl import flags
from flax import nnx
from PIL import Image
import optax

from src.data.fashion_mnist import get_fashion_mnist_dataset
from src.models.dit.dit import DiT
from src.training.ema import EMAManager
from src.training.sampler import DDIMSampler
from src.training.step import compute_loss, train_step
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.logging import TensorBoardLogger
from src.utils.rng import RNGManager


FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/fashion_mnist.yaml", "Path to the config file.")
flags.DEFINE_string("output_dir", "./outputs/fashion_mnist", "Directory for logs and checkpoints.")


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
        grid.paste(Image.fromarray(image, mode=mode), ((index % columns) * width, (index // columns) * height))

    grid.save(path)


def main(_):
    """Train a class-conditional raw-pixel diffusion model."""
    config = load_config(FLAGS.config)
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    rng = RNGManager(config.training.seed)
    logger = TensorBoardLogger(os.path.join(FLAGS.output_dir, "logs"))
    checkpointer = CheckpointManager(
        os.path.join(FLAGS.output_dir, "checkpoints"),
        gcs_directory="gs://diffjax/models",
        artifact_paths=[
            os.path.join(FLAGS.output_dir, "logs"),
            os.path.join(FLAGS.output_dir, "train_logs.txt"),
        ],
    )
    sample_artifact_dir = os.path.join(FLAGS.output_dir, "checkpoints", "samples")
    os.makedirs(sample_artifact_dir, exist_ok=True)

    model = DiT(
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
        learn_sigma=config.model.get("learn_sigma", False),
        rngs=nnx.Rngs(rng.next()),
    )
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
    dataset_iter = iter(dataset)
    validation_dataset = get_fashion_mnist_dataset(
        batch_size=config.training.batch_size,
        split="test",
        shuffle=False,
        seed=config.training.seed,
        dataset_name=config.data.dataset_name,
    )
    validation_iter = iter(validation_dataset)

    @nnx.jit
    def validation_step(model, latents, labels, rng_key):
        return compute_loss(model, latents, labels, rng_key)

    trainlog_path = os.path.join(FLAGS.output_dir, "train_logs.txt")
    trainlog = open(trainlog_path, "w", encoding="utf-8")

    def emit(message: str) -> None:
        """Write the same human-readable message to the terminal and log file."""
        print(message, flush=True)
        trainlog.write(message + "\n")
        trainlog.flush()

    emit(f"Using device: {jax.devices()[0]}")
    emit("Training directly on 28x28x1 pixels; no VAE is used.")
    emit(f"Starting training for {config.training.total_steps} steps...")
    training_start = time.perf_counter()

    try:
        for step in range(config.training.total_steps):
            batch = next(dataset_iter)
            train_step_start = time.perf_counter()
            metrics = train_step(
                model,
                optimizer,
                batch["image"],
                batch["label"],
                rng.next(),
                use_bf16=False,
            )
            ema.update(model)

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
                        "performance/validation_step_sec": validation_step_duration,
                        "performance/validation_steps_per_sec": validation_steps_per_sec,
                        "performance/validation_samples_per_sec": validation_samples_per_sec,
                    },
                )
                logger.flush()
                checkpointer.sync_to_gcs()
                emit(
                    f"step={step + 1:05d}/{config.training.total_steps:05d} "
                    f"progress={progress_percent:.2f}% train_loss={loss:.6f} "
                    f"validation_loss={validation_loss:.6f} "
                    f"train_step_sec={train_step_duration:.6f} "
                    f"train_steps_per_sec={train_steps_per_sec:.6f} "
                    f"train_samples_per_sec={train_samples_per_sec:.6f} "
                    f"validation_step_sec={validation_step_duration:.6f} "
                    f"validation_steps_per_sec={validation_steps_per_sec:.6f} "
                    f"validation_samples_per_sec={validation_samples_per_sec:.6f} "
                    f"elapsed_sec={elapsed_sec:.6f}"
                )

            if step % config.evaluation.sampling_interval == 0:
                nnx.update(sampling_model, ema.ema_state)
                sample_count = config.evaluation.sample_count
                labels = jnp.arange(sample_count, dtype=jnp.int32) % config.model.num_classes
                null_labels = jnp.full_like(labels, config.model.num_classes)
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
                    cfg_scale=config.evaluation.cfg_scale,
                )
                sample_images = (samples + 1.0).clip(0.0, 2.0) / 2.0
                logger.log_images(step, "train/samples", sample_images)
                _save_sample_grid(
                    sample_images,
                    os.path.join(sample_artifact_dir, f"samples_step_{step:06d}.png"),
                )
                logger.flush()
                checkpointer.sync_to_gcs()

            if step % config.evaluation.checkpoint_interval == 0 and step > 0:
                checkpointer.save(
                    step,
                    {"model": nnx.state(model), "ema": ema.ema_state, "opt": nnx.state(optimizer)},
                )
        emit("Training complete.")
    finally:
        logger.close()
        trainlog.close()
