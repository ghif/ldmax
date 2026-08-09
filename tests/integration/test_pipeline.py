"""Integration tests for the full DiT pipeline."""

import os
from absl import flags

from src.training.runner import main as train_main

def test_full_pipeline_short_run(tmp_path):
    """Test that the full training pipeline can run for a few steps."""
    output_dir = tmp_path / "outputs"
    config_path = "configs/cifar10_test.yaml"
    
    # Use flags.FLAGS to mock CLI arguments
    FLAGS = flags.FLAGS
    # Parse flags manually for testing
    FLAGS(["test", "--config", config_path, "--output_dir", str(output_dir)])
    
    try:
        train_main([])
    except SystemExit:
        pass # absl.app.run might call sys.exit
        
    assert os.path.exists(output_dir / "checkpoints")
    assert os.path.exists(output_dir / "logs")
    # The test config runs for two steps, numbered 0 and 1.
    assert os.path.exists(output_dir / "checkpoints" / "1")
