"""Unit tests for the CIFAR-10 data pipeline."""

import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
from src.data.cifar import get_cifar10_dataset

def test_cifar10_loader_shapes():
    """Test that the CIFAR-10 loader returns expected batch shapes."""
    batch_size = 4
    loader = get_cifar10_dataset(batch_size=batch_size, shuffle=False)
    
    # Get first batch
    batch = next(iter(loader))
    
    assert "image" in batch
    assert "label" in batch
    assert batch["image"].shape == (batch_size, 32, 32, 3)
    assert batch["label"].shape == (batch_size,)
    
    # Check range [-1, 1]
    assert np.min(batch["image"]) >= -1.1
    assert np.max(batch["image"]) <= 1.1

def test_cifar10_loader_resizing():
    """Test that the CIFAR-10 loader correctly resizes images."""
    batch_size = 2
    target_size = 64
    loader = get_cifar10_dataset(batch_size=batch_size, shuffle=False, target_size=target_size)
    
    batch = next(iter(loader))
    assert batch["image"].shape == (batch_size, target_size, target_size, 3)

def test_visualize_cifar10():
    """Load a batch of CIFAR-10 images and save them to a file for manual verification."""
    batch_size = 16
    loader = get_cifar10_dataset(batch_size=batch_size, shuffle=True, seed=42)
    
    # Get first batch
    batch = next(iter(loader))
    images = batch["image"] # Shape (B, 32, 32, 3), range [-1, 1]
    labels = batch["label"]
    
    # Denormalize: [-1, 1] -> [0, 1]
    images = (images + 1.0) / 2.0
    images = np.clip(images, 0, 1)
    
    # Create a grid
    grid_size = int(np.sqrt(batch_size))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i, ax in enumerate(axes.flat):
        if i < batch_size:
            ax.imshow(images[i])
            ax.set_title(f"{classes[labels[i]]}")
        ax.axis("off")
    
    # Ensure output directory exists
    output_dir = "tests/debug_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "cifar10_samples.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nCIFAR-10 Visualization saved to: {output_path}")
    assert os.path.exists(output_path)
