"""Grain-based data pipeline for CelebA with 40 binary attribute labels."""

import grain.python as grain
import numpy as np
from typing import Mapping, Any
from datasets import load_dataset
from PIL import Image

CELEBA_ATTRIBUTE_NAMES = (
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
)

class CelebADataSource:
    """A Grain-compatible data source wrapping a Hugging Face CelebA dataset."""
    CELEBA_ATTRIBUTE_NAMES = CELEBA_ATTRIBUTE_NAMES
    
    def __init__(self, dataset, target_size: int = 64):
        self.dataset = dataset
        self.target_size = target_size
        
    def __len__(self):
        return len(self.dataset)

    def _extract_label(self, item: Mapping[str, Any]) -> np.ndarray:
        """Extract the 40 CelebA attribute labels as a binary vector."""
        attributes = item.get("attributes")
        if isinstance(attributes, Mapping):
            return np.array([np.int32(bool(attributes[name])) for name in self.CELEBA_ATTRIBUTE_NAMES], dtype=np.int32)

        if isinstance(attributes, (list, tuple, np.ndarray)):
            attrs = np.asarray(attributes)
            if attrs.ndim == 1 and attrs.shape[0] >= len(self.CELEBA_ATTRIBUTE_NAMES):
                attrs = attrs[: len(self.CELEBA_ATTRIBUTE_NAMES)]
                return attrs.astype(np.int32)

        return np.array([np.int32(bool(item[name])) for name in self.CELEBA_ATTRIBUTE_NAMES], dtype=np.int32)
        
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
        
        return {
            "image": img_normalized,
            "label": self._extract_label(item)
        }

def get_celeba_dataset(
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 0,
    target_size: int = 64,
    dataset_name: str = "flwrlabs/celeba",
    dataset_config: str = "img_align+identity+attr",
) -> grain.DataLoader:
    """Initialize the CelebA data loader using Grain.

    Args:
        batch_size: Number of elements per batch.
        split: Dataset split ('train' or 'test').
        shuffle: Whether to shuffle the data.
        seed: Random seed for shuffling.
        target_size: The resolution to resize images to.
        dataset_name: Hugging Face dataset source that includes the 40 CelebA attributes.
        dataset_config: Dataset config/subset for the selected CelebA source.

    Returns:
        A Grain DataLoader instance.
    """
    # Load CelebA dataset from Hugging Face
    hf_dataset = load_dataset(dataset_name, dataset_config, split=split)
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
