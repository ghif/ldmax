"""Tests for Fashion MNIST checkpoint-resume path handling."""

from pathlib import Path

import pytest

pytest.importorskip("datasets")

from src.training.fashion_mnist_runner import _resolve_resume_checkpoint


def test_resolve_run_directory_uses_latest_checkpoint(tmp_path, monkeypatch):
    """A run directory resolves to its checkpoint root and latest step."""
    run_dir = tmp_path / "fashion_run"
    (run_dir / "checkpoints").mkdir(parents=True)

    class FakeCheckpointManager:
        def __init__(self, directory):
            assert Path(directory) == run_dir / "checkpoints"

        def latest_step(self):
            return 5000

    monkeypatch.setattr(
        "src.training.fashion_mnist_runner.CheckpointManager",
        FakeCheckpointManager,
    )

    checkpoint_root, step = _resolve_resume_checkpoint(str(run_dir))

    assert checkpoint_root == (run_dir / "checkpoints").resolve()
    assert step == 5000


def test_resolve_individual_checkpoint(tmp_path):
    """An individual Orbax checkpoint resolves without scanning siblings."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "5000"
    (checkpoint / "default").mkdir(parents=True)

    resolved_root, step = _resolve_resume_checkpoint(str(checkpoint))

    assert resolved_root == checkpoint_root.resolve()
    assert step == 5000


def test_resolve_rejects_unrecognized_path(tmp_path):
    """Invalid resume paths fail before model construction."""
    with pytest.raises(ValueError, match="--resume_from"):
        _resolve_resume_checkpoint(str(tmp_path))
