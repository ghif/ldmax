"""Integration tests for the full DiT pipeline."""

import os
import shutil
import pytest
from absl import flags
import jax.numpy as jnp

from src.scripts.train import main as train_main

def test_full_pipeline_short_run(tmp_path):
    """Test that the full training pipeline can run for a few steps."""
    output_dir = tmp_path / "outputs"
    config_path = "configs/cifar10.yaml"
    
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
    # Check if a checkpoint was saved (step 5 and last step 9)
    assert os.path.exists(output_dir / "checkpoints" / "9")
