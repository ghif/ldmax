"""Unit tests for src/training/trainer.py."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp

from src.data.factory import DataLoaderBundle, DatasetMetadata
from src.models.dit.dit import DiT
from src.training.trainer import Trainer, validation_step


def test_validation_step():
    """Verify validation step computes scalar MSE loss without gradient computation."""
    rng = jax.random.key(0)
    rng_model, rng_key = jax.random.split(rng)

    from flax import nnx

    model = DiT(
        input_size=8,
        patch_size=2,
        in_channels=1,
        hidden_size=32,
        depth=1,
        num_heads=2,
        num_classes=10,
        label_mode="class",
        rngs=nnx.Rngs(rng_model),
    )

    dummy_inputs = jnp.zeros((2, 8, 8, 1), dtype=jnp.float32)
    dummy_labels = jnp.array([0, 1], dtype=jnp.int32)

    val_loss = validation_step(model, dummy_inputs, dummy_labels, rng_key)
    assert val_loss.ndim == 0
    assert float(val_loss) >= 0.0


def test_trainer_initialization_and_short_run(tmp_path, monkeypatch):
    """Verify Trainer initializes components and executes training steps."""
    config = SimpleNamespace(
        dataset="test_data",
        model=SimpleNamespace(
            type="dit",
            input_size=8,
            patch_size=2,
            in_channels=1,
            hidden_size=32,
            depth=1,
            num_heads=2,
            num_classes=10,
            label_mode="class",
            label_dim=None,
            learn_sigma=False,
        ),
        training=SimpleNamespace(
            learning_rate=0.001,
            weight_decay=0.01,
            batch_size=2,
            total_steps=2,
            seed=42,
            ema_decay=0.99,
            use_bf16=False,
            prefetch_size=0,
            gcs_directory="",
        ),
        evaluation=SimpleNamespace(
            log_interval=1,
            checkpoint_interval=2,
            sampling_interval=2,
            sample_count=2,
            num_inference_steps=2,
            cfg_scale=1.0,
        ),
    )

    metadata = DatasetMetadata(
        name="test_data",
        is_latent=False,
        image_size=8,
        channels=1,
        num_classes=10,
        label_mode="class",
    )

    def dummy_iterator():
        while True:
            yield {
                "image": jnp.zeros((2, 8, 8, 1), dtype=jnp.float32),
                "label": jnp.array([1, 2], dtype=jnp.int32),
            }

    fake_bundle = DataLoaderBundle(
        train_iter=dummy_iterator(),
        val_iter=dummy_iterator(),
        metadata=metadata,
    )

    monkeypatch.setattr("src.training.trainer.create_dataloaders", lambda cfg: fake_bundle)

    output_dir = tmp_path / "trainer_run"
    trainer = Trainer(
        config=config,
        output_dir=output_dir,
    )

    assert trainer.start_step == 0
    assert (output_dir / "logs").is_dir()

    # Run the 2 steps
    trainer.run()

    # Verify artifacts and checkpoint directories
    assert (output_dir / "checkpoints").is_dir()

    # Verify train_logs.txt exists and captured detailed telemetry
    log_file = output_dir / "train_logs.txt"
    assert log_file.is_file()
    log_content = log_file.read_text(encoding="utf-8")
    assert "step=00001/00002" in log_content
    assert "train_step_sec=" in log_content
    assert "train_samples_per_sec=" in log_content
    assert "data_wait_sec=" in log_content
    assert "Training finished in" in log_content
