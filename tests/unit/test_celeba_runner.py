"""Unit tests for the CelebA Latent VAE runner."""

from pathlib import Path
import jax.numpy as jnp
import pytest
from flax import nnx

from src.models.dit.dit import DiT
from src.training.celeba_runner import (
    _build_model,
    _checkpoint_state,
    _resolve_resume_checkpoint,
    _save_sample_grid,
    _validate_config,
)
from src.utils.config import load_config


def test_celeba_config_validation():
    """Verify that configuration invariants are enforced."""
    config = load_config("configs/celeba.yaml")
    _validate_config(config)

    # Test invalid input size
    config.model.input_size = 16
    with pytest.raises(ValueError, match="model.input_size=32"):
        _validate_config(config)

    config.model.input_size = 32
    config.model.in_channels = 3
    with pytest.raises(ValueError, match="model.in_channels=4"):
        _validate_config(config)

    config.model.in_channels = 4
    config.model.label_mode = "class"
    with pytest.raises(ValueError, match="model.label_mode='attributes'"):
        _validate_config(config)

    config.model.label_mode = "attributes"
    config.model.label_dim = 10
    with pytest.raises(ValueError, match="model.label_dim=40"):
        _validate_config(config)

    config.model.label_dim = 40
    config.data.image_size = 128
    with pytest.raises(ValueError, match="data.image_size=256"):
        _validate_config(config)


def test_celeba_build_model_and_forward():
    """Verify DiT model construction with attribute conditioning on latents."""
    config = load_config("configs/celeba.yaml")
    config.model.hidden_size = 64
    config.model.depth = 2
    config.model.num_heads = 2
    config.training.use_bf16 = False

    rng_key = jnp.array([0, 1], dtype=jnp.uint32)
    model = _build_model(config, rng_key)

    batch_size = 2
    dummy_latents = jnp.zeros((batch_size, 32, 32, 4), dtype=jnp.float32)
    dummy_timesteps = jnp.array([10, 200], dtype=jnp.int32)
    dummy_labels = jnp.ones((batch_size, 40), dtype=jnp.int32)

    output = model(dummy_latents, dummy_timesteps, dummy_labels)
    assert output.shape == (batch_size, 32, 32, 4)


def test_celeba_checkpoint_state_conversion():
    """Ensure floating leaves are converted to float32."""
    state = {
        "params": {
            "w": jnp.zeros((2, 2), dtype=jnp.bfloat16),
            "b": jnp.zeros((2,), dtype=jnp.float32),
            "step": 10,
        }
    }
    converted = _checkpoint_state(state)
    assert converted["params"]["w"].dtype == jnp.float32
    assert converted["params"]["b"].dtype == jnp.float32
    assert converted["params"]["step"] == 10


def test_celeba_resolve_resume_checkpoint(tmp_path, monkeypatch):
    """Verify run directory and checkpoint directory path resolution."""
    run_dir = tmp_path / "celeba_run"
    (run_dir / "checkpoints").mkdir(parents=True)

    class FakeCheckpointManager:
        def __init__(self, directory):
            assert Path(directory) == run_dir / "checkpoints"

        def latest_step(self):
            return 10000

    monkeypatch.setattr(
        "src.training.celeba_runner.CheckpointManager",
        FakeCheckpointManager,
    )

    checkpoint_root, step = _resolve_resume_checkpoint(str(run_dir))
    assert checkpoint_root == (run_dir / "checkpoints").resolve()
    assert step == 10000

    individual = run_dir / "checkpoints" / "10000"
    (individual / "default").mkdir(parents=True)
    resolved_root, step_ind = _resolve_resume_checkpoint(str(individual))
    assert resolved_root == (run_dir / "checkpoints").resolve()
    assert step_ind == 10000

    # Nonexistent path
    with pytest.raises(FileNotFoundError, match="Resume path does not exist"):
        _resolve_resume_checkpoint(str(tmp_path / "nonexistent"))

    # Unrecognized directory format
    dummy_dir = tmp_path / "other_dir"
    dummy_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="--resume_from"):
        _resolve_resume_checkpoint(str(dummy_dir))


def test_celeba_save_sample_grid(tmp_path):
    """Verify sample grid PNG saving from [0, 1] RGB batch."""
    dummy_samples = jnp.zeros((4, 64, 64, 3), dtype=jnp.float32)
    output_path = tmp_path / "samples.png"
    _save_sample_grid(dummy_samples, str(output_path))
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
