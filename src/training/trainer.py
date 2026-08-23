"""Unified Trainer engine for Diffusion Transformer (DiT) models."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import optax
from flax import nnx

from src.data.factory import DataLoaderBundle, create_dataloaders
from src.models.factory import create_model
from src.training.checkpointing import (
    checkpoint_has_rng,
    checkpoint_state,
    resolve_resume_checkpoint,
    restore_args,
    restore_template,
    restore_training_state,
)
from src.training.ema import EMAManager
from src.training.evaluator import Evaluator
from src.training.step import compute_loss, train_step
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.logging import TensorBoardLogger
from src.utils.rng import RNGManager
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


@nnx.jit
def validation_step(
    model: nnx.Module,
    latents_or_pixels: jax.Array,
    labels: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Compute diffusion loss on validation batch with dropout disabled."""
    return compute_loss(model, latents_or_pixels, labels, key, train=False)


class Trainer:
    """Config-driven training controller for LDMAX diffusion models."""

    def __init__(
        self,
        config: str | Path | Any,
        output_dir: str | Path = "",
        resume_from: str | Path = "",
    ):
        """Initialize the Trainer.

        Args:
            config: Path to YAML config file or loaded config object.
            output_dir: Optional output directory for logs and checkpoints.
            resume_from: Optional run directory or checkpoint step to resume from.
        """
        if isinstance(config, (str, Path)):
            self.config = load_config(str(config))
        else:
            self.config = config

        self.dataset_name = _cfg_get(self.config, "dataset", "cifar10")
        if not output_dir:
            output_dir = f"./outputs/{self.dataset_name}"
        self.output_dir = str(Path(output_dir).expanduser().resolve())
        os.makedirs(self.output_dir, exist_ok=True)

        self.resume_from = str(resume_from).strip() if resume_from else ""

        # Configure mixed precision
        self.use_bf16 = bool(
            _cfg_get(self.config.training, "use_bf16", False)
            or _cfg_get(self.config.training, "mixed_precision", False)
        )
        if self.use_bf16:
            jax.config.update("jax_default_matmul_precision", "bfloat16")

        # Initialize RNG, Logging, Checkpointing
        seed = self.config.training.seed
        self.rng = RNGManager(seed)
        self.logger = TensorBoardLogger(os.path.join(self.output_dir, "logs"))

        log_path = os.path.join(self.output_dir, "train_logs.txt")
        self.train_log_file = open(log_path, "a" if self.resume_from else "w", encoding="utf-8")

        gcs_dir = _cfg_get(self.config.training, "gcs_directory", "gs://diffjax/models")
        if not gcs_dir or not str(gcs_dir).startswith("gs://"):
            gcs_dir = None

        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.checkpointer = CheckpointManager(
            checkpoint_dir,
            gcs_directory=gcs_dir,
            best_metric="validation_loss",
            best_mode="min",
            artifact_paths=[
                os.path.join(self.output_dir, "logs"),
                os.path.join(self.output_dir, "train_logs.txt"),
            ],
        )

        # Setup Data Pipelines
        self.data_bundle: DataLoaderBundle = create_dataloaders(self.config)
        self.metadata = self.data_bundle.metadata

        # Setup Models and Optimizer
        self.model = create_model(self.config, self.rng.next())
        self.optimizer = nnx.Optimizer(
            self.model,
            optax.adamw(
                learning_rate=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            ),
            wrt=nnx.Param,
        )
        self.ema = EMAManager(self.model, decay=self.config.training.ema_decay)
        self.sampling_model = create_model(self.config, self.rng.next())

        # Setup VAE for latent datasets
        if self.metadata.is_latent:
            self.vae_manager = VAEManager()

            @jax.jit
            def _encode(images: jax.Array, key: jax.Array) -> jax.Array:
                return self.vae_manager.encode(images, key)

            self.encode_fn = _encode
        else:
            self.vae_manager = None
            self.encode_fn = None

        self.evaluator = Evaluator(
            self.config,
            self.metadata,
            vae_manager=self.vae_manager,
        )

        self.start_step = 0
        if self.resume_from:
            self._resume_checkpoint()

    def emit(self, message: str) -> None:
        """Print message to stdout and mirror to train_logs.txt."""
        print(message, flush=True)
        if (
            hasattr(self, "train_log_file")
            and self.train_log_file
            and not self.train_log_file.closed
        ):
            self.train_log_file.write(message + "\n")
            self.train_log_file.flush()

    def _resume_checkpoint(self) -> None:
        """Restore model, optimizer, EMA, and RNG state from resume path."""
        checkpoint_root, restored_step = resolve_resume_checkpoint(self.resume_from)
        has_rng = checkpoint_has_rng(checkpoint_root, restored_step)

        source_checkpointer = CheckpointManager(str(checkpoint_root))
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        template = {
            "model": restore_template(nnx.state(self.model), sharding),
            "ema": restore_template(self.ema.ema_state, sharding),
            "opt": restore_template(nnx.state(self.optimizer), sharding),
        }
        if has_rng:
            template["rng"] = {"key": jax.device_put(self.rng.state, sharding)}

        checkpoint_state_dict = source_checkpointer.restore(
            restored_step,
            args=restore_args(template, sharding),
        )
        if checkpoint_state_dict is None:
            raise ValueError(f"Unable to restore checkpoint step {restored_step}")

        conditioning = _cfg_get(self.config.model, "conditioning", None)
        restore_training_state(
            self.model,
            self.optimizer,
            self.ema,
            checkpoint_state_dict,
            conditioning=conditioning,
        )

        rng_state = checkpoint_state_dict.get("rng")
        if isinstance(rng_state, Mapping) and "key" in rng_state:
            self.rng.restore(rng_state["key"])
        else:
            self.rng = RNGManager.from_seed_and_step(self.config.training.seed, restored_step)

        self.start_step = restored_step
        self.emit(f"Resumed training from step {restored_step} ({checkpoint_root})")

    def run(self) -> None:
        """Execute the training loop."""
        total_steps = self.config.training.total_steps
        eval_cfg = getattr(self.config, "evaluation", None)
        log_interval = _cfg_get(eval_cfg, "log_interval", 100) if eval_cfg else 100
        checkpoint_interval = _cfg_get(eval_cfg, "checkpoint_interval", 5000) if eval_cfg else 5000
        sampling_interval = _cfg_get(eval_cfg, "sampling_interval", 1000) if eval_cfg else 1000

        data_iter = self.data_bundle.train_iter
        val_iter = self.data_bundle.val_iter

        training_start = time.perf_counter()
        latest_val_loss = float("inf")
        sampling_duration = 0.0
        gcs_sync_duration = 0.0

        try:
            for step in range(self.start_step, total_steps):
                data_wait_start = time.perf_counter()
                batch = next(data_iter)
                data_wait_duration = time.perf_counter() - data_wait_start

                if self.encode_fn is not None:
                    inputs = self.encode_fn(batch["image"], self.rng.next())
                else:
                    inputs = batch["image"]

                labels = batch.get("label")

                train_step_start = time.perf_counter()
                metrics = train_step(
                    self.model,
                    self.optimizer,
                    inputs,
                    labels,
                    self.rng.next(),
                    use_bf16=self.use_bf16,
                )
                train_dispatch_duration = time.perf_counter() - train_step_start

                ema_start = time.perf_counter()
                self.ema.update(self.model)
                ema_update_duration = time.perf_counter() - ema_start

                # Periodic Evaluation & Validation Logging
                if step % log_interval == 0:
                    train_loss = float(metrics["loss"])
                    train_step_duration = time.perf_counter() - train_step_start
                    train_steps_per_sec = 1.0 / max(train_step_duration, 1e-12)
                    batch_size = inputs.shape[0]
                    train_samples_per_sec = batch_size * train_steps_per_sec
                    elapsed_sec = time.perf_counter() - training_start

                    if val_iter is not None:
                        val_batch = next(val_iter)
                        val_inputs = (
                            self.encode_fn(val_batch["image"], self.rng.next())
                            if self.encode_fn is not None
                            else val_batch["image"]
                        )
                        val_labels = val_batch.get("label")
                        val_start = time.perf_counter()
                        val_loss = float(
                            validation_step(
                                self.model,
                                val_inputs,
                                val_labels,
                                self.rng.next(),
                            )
                        )
                        val_step_duration = time.perf_counter() - val_start
                        latest_val_loss = val_loss
                    else:
                        val_loss = 0.0
                        val_step_duration = 0.0

                    progress = 100.0 * (step + 1) / total_steps
                    self.logger.log_scalars(
                        step,
                        {
                            "train/loss": train_loss,
                            "validation/loss": val_loss,
                            "performance/train_step_sec": train_step_duration,
                            "performance/train_steps_per_sec": train_steps_per_sec,
                            "performance/train_samples_per_sec": train_samples_per_sec,
                            "performance/data_wait_sec": data_wait_duration,
                            "performance/train_dispatch_sec": train_dispatch_duration,
                            "performance/ema_update_sec": ema_update_duration,
                            "performance/validation_step_sec": val_step_duration,
                            "performance/sampling_sec": sampling_duration,
                            "performance/gcs_sync_sec": gcs_sync_duration,
                            "performance/elapsed_sec": elapsed_sec,
                        },
                    )
                    self.logger.flush()

                    self.emit(
                        f"step={step + 1:05d}/{total_steps:05d} "
                        f"progress={progress:.2f}% "
                        f"train_loss={train_loss:.6f} "
                        f"validation_loss={val_loss:.6f} "
                        f"train_step_sec={train_step_duration:.6f} "
                        f"train_steps_per_sec={train_steps_per_sec:.2f} "
                        f"train_samples_per_sec={train_samples_per_sec:.1f} "
                        f"data_wait_sec={data_wait_duration:.6f} "
                        f"train_dispatch_sec={train_dispatch_duration:.6f} "
                        f"ema_update_sec={ema_update_duration:.6f} "
                        f"validation_step_sec={val_step_duration:.6f} "
                        f"sampling_sec={sampling_duration:.6f} "
                        f"gcs_sync_sec={gcs_sync_duration:.6f} "
                        f"elapsed_sec={elapsed_sec:.3f}"
                    )

                # Periodic Sampling
                if step % sampling_interval == 0:
                    _, sampling_duration = self.evaluator.evaluate_and_log_samples(
                        sampling_model=self.sampling_model,
                        ema_state=self.ema.ema_state,
                        batch=batch,
                        rng_key=self.rng.next(),
                        step=step,
                        logger=self.logger,
                        output_dir=self.output_dir,
                    )

                # Periodic Checkpointing
                should_checkpoint = (
                    step % checkpoint_interval == 0 and step > 0
                ) or step == total_steps - 1
                if should_checkpoint:
                    gcs_sync_start = time.perf_counter()
                    self.checkpointer.save(
                        step,
                        {
                            "model": checkpoint_state(nnx.state(self.model)),
                            "ema": checkpoint_state(self.ema.ema_state),
                            "opt": checkpoint_state(nnx.state(self.optimizer)),
                            "rng": {"key": self.rng.state},
                        },
                        metrics={"validation_loss": latest_val_loss},
                    )
                    gcs_sync_duration = time.perf_counter() - gcs_sync_start

        finally:
            self.logger.close()
            elapsed = time.perf_counter() - training_start
            self.emit(f"Training finished in {elapsed:.2f}s.")
            if (
                hasattr(self, "train_log_file")
                and self.train_log_file
                and not self.train_log_file.closed
            ):
                self.train_log_file.close()
            if self.checkpointer.gcs_directory is not None:
                self.checkpointer.sync_to_gcs()
