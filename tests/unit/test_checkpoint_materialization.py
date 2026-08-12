"""Tests for automatic checkpoint selection."""

from src.utils.checkpoint import _latest_gcs_step, _latest_local_checkpoint


def test_latest_local_checkpoint_selects_highest_step(tmp_path):
    """The highest numeric local checkpoint is selected."""
    for step in (1000, 5000, 3000):
        (tmp_path / str(step)).mkdir()
    (tmp_path / "samples").mkdir()

    assert _latest_local_checkpoint(tmp_path) == tmp_path / "5000"


def test_latest_gcs_step_selects_highest_immediate_step():
    """Only the first numeric directory below the GCS prefix is considered."""
    names = [
        "models/run/checkpoints/1000/default/_METADATA",
        "models/run/checkpoints/5000/default/_METADATA",
        "models/run/checkpoints/5000/samples/sample.png",
        "models/run/checkpoints/samples/sample.png",
    ]

    assert _latest_gcs_step(names, "models/run/checkpoints") == 5000
