"""Grain-based data pipeline for CelebA."""

import grain.python as grain
import numpy as np
from typing import Mapping, Any
from datasets import load_dataset
from PIL import Image

class CelebADataSource:
    """A Grain-compatible data source wrapping a Hugging Face CelebA dataset."""
    
    def __init__(self, dataset, target_size: int = 64):
        self.dataset = dataset
        self.target_size = target_size
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['image'] # HF 'nielsr/CelebA-faces' has 'image' column
        
        # Center crop and resize
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = (w + crop_size) // 2
        bottom = (h + crop_size) // 2
        
        img = img.crop((left, top, right, bottom))
        img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
        
        # Convert to numpy array (H, W, C), scale to [-1, 1]
        img_np = np.array(img, dtype=np.float32)
        img_normalized = (img_np / 127.5) - 1.0
        
        # CelebA labels are usually attributes, but if we just want a dummy for now:
        return {
            "image": img_normalized,
            "label": 0 # Default label for now
        }

def get_celeba_dataset(
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 0,
    target_size: int = 64
) -> grain.DataLoader:
    """Initialize the CelebA data loader using Grain.

    Args:
        batch_size: Number of elements per batch.
        split: Dataset split ('train' or 'test').
        shuffle: Whether to shuffle the data.
        seed: Random seed for shuffling.
        target_size: The resolution to resize images to.

    Returns:
        A Grain DataLoader instance.
    """
    # Load CelebA dataset from Hugging Face
    hf_dataset = load_dataset("nielsr/CelebA-faces", split=split)
    source = CelebADataSource(hf_dataset, target_size=target_size)
    
    transformations = [
        grain.Batch(batch_size=batch_size, drop_remainder=True)
    ]
    
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None,
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        shuffle=shuffle,
        seed=seed
    )
    
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=4,
        operations=transformations
    )
    
    return loader
