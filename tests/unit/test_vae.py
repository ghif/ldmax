"""Unit tests for the VAE manager."""

import os
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.utils.vae import VAEManager
from src.data.cifar import get_cifar10_dataset
from src.data.celeba import get_celeba_dataset

@pytest.fixture(scope="module")
def vae_manager():
    """Fixture to load VAEManager once for all tests."""
    return VAEManager()

def test_vae_shapes(vae_manager):
    """Test that VAE produces correct shapes for latents and reconstructed images."""
    batch_size = 1
    image_size = 64 # Must be multiple of 8
    
    # Create dummy image in [-1, 1] range
    key = jax.random.PRNGKey(0)
    img_key, encode_key = jax.random.split(key)
    image = jax.random.uniform(img_key, (batch_size, image_size, image_size, 3)) * 2 - 1
    
    # Encode
    latents = vae_manager.encode(image, encode_key)
    
    # Expected latent shape is (B, H/8, W/8, 4) if NHWC
    expected_latent_size = image_size // 8
    assert latents.shape == (batch_size, expected_latent_size, expected_latent_size, 4)
    
    # Decode
    reconstructed = vae_manager.decode(latents)
    
    # Reconstructed should be in [0, 1] range
    assert reconstructed.shape == (batch_size, image_size, image_size, 3)
    assert jnp.all(reconstructed >= 0)
    assert jnp.all(reconstructed <= 1)

def test_vae_reconstruction(vae_manager):
    """Test that VAE can reconstruct an image with reasonable accuracy."""
    batch_size = 1
    image_size = 32
    
    # Create a simple image (e.g., a solid color or a gradient)
    image = jnp.ones((batch_size, image_size, image_size, 3)) * 0.5 # mid-gray in [0, 1]
    image_norm = image * 2 - 1 # map to [-1, 1]
    
    key = jax.random.PRNGKey(42)
    latents = vae_manager.encode(image_norm, key)
    reconstructed = vae_manager.decode(latents)
    
    # Reconstructed should be close to original image (which was 0.5)
    mse = jnp.mean((reconstructed - image) ** 2)
    # VAE is lossy, but for a simple constant image it should be fairly accurate.
    # Note: CIFAR-10 size (32x32) is very small for SD VAE, but should still work.
    assert mse < 0.05

def test_vae_cifar10_visualization(vae_manager):
    """Test VAE encoding/decoding on real CIFAR-10 data and save visualization.
    
    Targeting 16x16 latent space requires 128x128 input images (8x downsampling).
    """
    batch_size = 4
    target_size = 128 # 16 * 8
    loader = get_cifar10_dataset(batch_size=batch_size, shuffle=True, seed=42, target_size=target_size)
    
    # Get a batch
    batch = next(iter(loader))
    images = batch["image"] # (B, 128, 128, 3), range [-1, 1]
    
    # Encode
    key = jax.random.PRNGKey(0)
    latents = vae_manager.encode(images, key) # (B, 16, 16, 4)
    
    assert latents.shape == (batch_size, 16, 16, 4), f"Expected (B, 16, 16, 4), got {latents.shape}"
    
    # Decode
    reconstructed = vae_manager.decode(latents) # (B, 128, 128, 3), range [0, 1]
    
    # Process for visualization
    # 1. Original images: [-1, 1] -> [0, 1]
    images_viz = (images + 1.0) / 2.0
    images_viz = np.clip(images_viz, 0, 1)
    
    # 2. Latents: Show first 3 channels as RGB, normalized for visibility
    # Shape is (B, 4, 4, 4)
    latents_viz = np.array(latents[:, :, :, :3])
    # Normalize latents for visualization
    latents_viz = (latents_viz - latents_viz.min()) / (latents_viz.max() - latents_viz.min() + 1e-8)
    
    # 3. Reconstructed: already [0, 1]
    reconstructed_viz = np.array(reconstructed)
    
    # Create comparison grid
    num_rows = batch_size
    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
    
    cols = ["Original", "Latent (RGB slice)", "Reconstructed"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
        
    for i in range(num_rows):
        axes[i, 0].imshow(images_viz[i])
        axes[i, 1].imshow(latents_viz[i])
        axes[i, 2].imshow(reconstructed_viz[i])
        
        for j in range(3):
            axes[i, j].axis("off")
            
    # Save to file
    output_dir = "tests/debug_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vae_cifar10_reconstruction.png")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nVAE CIFAR-10 visualization saved to: {output_path}")
    assert os.path.exists(output_path)

def test_vae_celeba_visualization(vae_manager):
    """Test VAE encoding/decoding on real CelebA data and save visualization.
    
    Targeting 32x32 latent space requires 256x256 input images (8x downsampling).
    """
    batch_size = 4
    target_size = 256 # 32 * 8
    loader = get_celeba_dataset(batch_size=batch_size, shuffle=True, seed=42, target_size=target_size)
    
    # Get a batch
    batch = next(iter(loader))
    images = batch["image"] # (B, 256, 256, 3), range [-1, 1]
    
    # Encode
    key = jax.random.PRNGKey(0)
    latents = vae_manager.encode(images, key) # (B, 32, 32, 4)
    
    assert latents.shape == (batch_size, 32, 32, 4), f"Expected (B, 32, 32, 4), got {latents.shape}"
    
    # Decode
    reconstructed = vae_manager.decode(latents) # (B, 256, 256, 3), range [0, 1]
    
    # Process for visualization
    # 1. Original images: [-1, 1] -> [0, 1]
    images_viz = (images + 1.0) / 2.0
    images_viz = np.clip(images_viz, 0, 1)
    
    # 2. Latents: Show first 3 channels as RGB, normalized for visibility
    latents_viz = np.array(latents[:, :, :, :3])
    latents_viz = (latents_viz - latents_viz.min()) / (latents_viz.max() - latents_viz.min() + 1e-8)
    
    # 3. Reconstructed: already [0, 1]
    reconstructed_viz = np.array(reconstructed)
    
    # Create comparison grid
    num_rows = batch_size
    fig, axes = plt.subplots(num_rows, 3, figsize=(10, 3 * num_rows))
    
    cols = ["Original", "Latent (RGB slice)", "Reconstructed"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)
        
    for i in range(num_rows):
        axes[i, 0].imshow(images_viz[i])
        axes[i, 1].imshow(latents_viz[i])
        axes[i, 2].imshow(reconstructed_viz[i])
        
        for j in range(3):
            axes[i, j].axis("off")
            
    # Save to file
    output_dir = "tests/debug_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vae_celeba_reconstruction.png")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"\nVAE CelebA visualization saved to: {output_path}")
    assert os.path.exists(output_path)
