"""Grain-based data pipeline for CIFAR-10."""

import grain.python as grain
import numpy as np
from typing import Mapping, Any, Optional
from datasets import load_dataset
from PIL import Image

class HFDataSource:
    """A Grain-compatible data source wrapping a Hugging Face dataset."""
    
    def __init__(self, dataset, target_size: Optional[int] = None):
        self.dataset = dataset
        self.target_size = target_size
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Hugging Face returns a PIL image for CIFAR-10 in 'img' column
        item = self.dataset[idx]
        img = item['img']
        
        # Optional resize
        if self.target_size is not None:
            img = img.resize((self.target_size, self.target_size), Image.LANCZOS)
            
        # Convert PIL Image to numpy array (H, W, C), scale to [-1, 1]
        img_np = np.array(img, dtype=np.float32)
        img_normalized = (img_np / 127.5) - 1.0
        
        return {
            "image": img_normalized,
            "label": np.int32(item['label'])
        }

def get_cifar10_dataset(
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 0,
    target_size: Optional[int] = None
) -> grain.DataLoader:
    """Initialize the CIFAR-10 data loader using Grain.

    Args:
        batch_size: Number of elements per batch.
        split: Dataset split ('train' or 'test').
        shuffle: Whether to shuffle the data.
        seed: Random seed for shuffling.
        target_size: Optional resolution to resize images to.

    Returns:
        A Grain DataLoader instance.
    """
    # Load real CIFAR-10 dataset from Hugging Face
    hf_dataset = load_dataset("uoft-cs/cifar10", split=split)
    source = HFDataSource(hf_dataset, target_size=target_size)
    
    transformations = [
        grain.Batch(batch_size=batch_size, drop_remainder=True)
    ]
    
    # Simple index-based sampler
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None,  # Infinite
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        shuffle=shuffle,
        seed=seed
    )
    
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=4,  # Use background workers for parallel loading
        operations=transformations
    )
    
    return loader
