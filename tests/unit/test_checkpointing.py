"""Unit tests for src/training/checkpointing.py."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.training.checkpointing import (
    checkpoint_has_rng,
    checkpoint_state,
    resolve_resume_checkpoint,
    restore_nnx_state,
    restore_template,
    validate_nnx_state,
)


def test_checkpoint_state_converts_floats_to_fp32():
    """Verify that floating-point arrays are cast to float32 while leaving ints intact."""
    state = {
        "params": {
            "w": jnp.zeros((2, 2), dtype=jnp.bfloat16),
            "b": jnp.zeros((2,), dtype=jnp.float16),
            "count": 5,
        }
    }
    converted = checkpoint_state(state)
    assert converted["params"]["w"].dtype == jnp.float32
    assert converted["params"]["b"].dtype == jnp.float32
    assert converted["params"]["count"] == 5


def test_resolve_resume_checkpoint_run_directory(tmp_path, monkeypatch):
    """Verify that providing a run directory resolves to its checkpoints/ child."""
    run_dir = tmp_path / "test_run"
    (run_dir / "checkpoints").mkdir(parents=True)

    class FakeCheckpointManager:
        """Mock CheckpointManager returning a fixed step."""

        def __init__(self, directory):
            assert Path(directory) == run_dir / "checkpoints"

        def latest_step(self):
            return 5000

    monkeypatch.setattr(
        "src.training.checkpointing.CheckpointManager",
        FakeCheckpointManager,
    )

    root, step = resolve_resume_checkpoint(str(run_dir))
    assert root == (run_dir / "checkpoints").resolve()
    assert step == 5000


def test_resolve_resume_checkpoint_individual_step(tmp_path):
    """Verify that an individual numeric step directory resolves to its parent root."""
    ckpt_root = tmp_path / "checkpoints"
    step_dir = ckpt_root / "2000"
    (step_dir / "default").mkdir(parents=True)

    root, step = resolve_resume_checkpoint(str(step_dir))
    assert root == ckpt_root.resolve()
    assert step == 2000


def test_resolve_resume_checkpoint_errors(tmp_path):
    """Verify error cases for missing paths and non-checkpoint directories."""
    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(str(tmp_path / "nonexistent"))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="--resume_from"):
        resolve_resume_checkpoint(str(empty_dir))


def test_checkpoint_has_rng(tmp_path):
    """Verify inspection of Orbax _METADATA for RNG keys."""
    ckpt_dir = tmp_path / "100" / "default"
    ckpt_dir.mkdir(parents=True)

    assert not checkpoint_has_rng(tmp_path, 100)

    metadata_path = ckpt_dir / "_METADATA"
    metadata_path.write_text(
        json.dumps({"tree_metadata": {"('rng', 'key')": {}}}),
        encoding="utf-8",
    )
    assert checkpoint_has_rng(tmp_path, 100)

    metadata_path.write_text(
        json.dumps({"tree_metadata": {"('model', 'weight')": {}}}),
        encoding="utf-8",
    )
    assert not checkpoint_has_rng(tmp_path, 100)


class SimpleLinear(nnx.Module):
    """Simple linear layer for testing NNX state validation and restoration."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        """Initialize linear layer parameters."""
        self.w = nnx.Param(jax.random.normal(rngs.params(), (in_features, out_features)))
        self.b = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x):
        """Perform linear transformation."""
        return x @ self.w.value + self.b.value


def test_nnx_state_validation_and_restoration():
    """Verify template building, validation, and in-place restoration of NNX states."""
    rngs = nnx.Rngs(0)
    model1 = SimpleLinear(4, 2, rngs=rngs)
    model2 = SimpleLinear(4, 2, rngs=rngs)

    template = restore_template(nnx.state(model1))
    assert "w" in template
    assert "b" in template

    # Artificially alter model2 parameters
    model2.w.value = jnp.ones_like(model2.w.value) * 9.0

    # Convert model1 state to checkpoint pure dict
    ckpt_dict = {
        "w": {"value": jnp.ones((4, 2), dtype=jnp.float32) * 3.0},
        "b": {"value": jnp.ones((2,), dtype=jnp.float32) * 1.5},
    }

    validate_nnx_state(nnx.state(model2), ckpt_dict, "model")
    restore_nnx_state(nnx.state(model2), ckpt_dict, "model")

    assert jnp.allclose(model2.w.value, 3.0)
    assert jnp.allclose(model2.b.value, 1.5)


def test_validate_nnx_state_mismatches():
    """Verify that shape mismatches and missing paths trigger descriptive ValueErrors."""
    rngs = nnx.Rngs(0)
    model = SimpleLinear(4, 2, rngs=rngs)

    # Missing path
    with pytest.raises(ValueError, match="missing state path"):
        validate_nnx_state(nnx.state(model), {"w": {"value": jnp.zeros((4, 2))}}, "model")

    # Shape mismatch
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_nnx_state(
            nnx.state(model),
            {"w": {"value": jnp.zeros((4, 3))}, "b": {"value": jnp.zeros((2,))}},
            "model",
        )
