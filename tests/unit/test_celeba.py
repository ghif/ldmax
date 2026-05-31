"""Unit tests for the CelebA data pipeline."""

import pytest
import numpy as np
import os
import matplotlib.pyplot as plt
from src.data.celeba import get_celeba_dataset

def test_celeba_loader_shapes():
    """Test that the CelebA loader returns expected batch shapes and ranges."""
    batch_size = 4
    target_size = 64
    loader = get_celeba_dataset(batch_size=batch_size, shuffle=False, target_size=target_size)
    
    # Get first batch
    batch = next(iter(loader))
    
    assert "image" in batch
    assert "label" in batch
    assert batch["image"].shape == (batch_size, target_size, target_size, 3)
    
    # Check range [-1, 1]
    assert np.min(batch["image"]) >= -1.1 # allowance for small float errors
    assert np.max(batch["image"]) <= 1.1

def test_visualize_celeba():
    """Load a batch of CelebA images and save them to a file for manual verification."""
    batch_size = 9
    target_size = 64
    loader = get_celeba_dataset(batch_size=batch_size, shuffle=True, seed=42, target_size=target_size)
    
    # Get first batch
    batch = next(iter(loader))
    images = batch["image"]
    
    # Denormalize: [-1, 1] -> [0, 1]
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0, 1)
    
    # Create a grid
    grid_size = 3
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(9, 9))
    
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            ax.imshow(images[i])
        ax.axis("off")
    
    # Ensure output directory exists
    output_dir = "tests/debug_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "celeba_samples.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nCelebA Visualization saved to: {output_path}")
    assert os.path.exists(output_path)
