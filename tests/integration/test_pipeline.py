"""Integration tests for the full DiT pipeline."""

from pathlib import Path

from src.training.trainer import Trainer


def test_full_pipeline_short_run(tmp_path):
    """Test that the full training pipeline can run for a few steps."""
    output_dir = tmp_path / "outputs"
    config_path = Path("configs/cifar10_test.yaml")

    trainer = Trainer(
        config=config_path,
        output_dir=output_dir,
    )
    trainer.run()

    assert (output_dir / "checkpoints").exists()
    assert (output_dir / "logs").exists()
    assert (output_dir / "checkpoints" / "1").exists()
