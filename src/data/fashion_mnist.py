"""Grain data pipeline for raw-pixel Fashion MNIST training."""

from typing import Any

import grain.python as grain
import numpy as np
from datasets import load_dataset


class FashionMNISTDataSource:
    """Adapt a Hugging Face Fashion MNIST split to the Grain protocol."""

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.int32]:
        item = self.dataset[index]
        image = np.asarray(item["image"], dtype=np.float32) / 127.5 - 1.0
        image = image[..., None]  # NHWC: 28 x 28 x 1
        return {"image": image, "label": np.int32(item["label"])}


def get_fashion_mnist_dataset(
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 0,
    dataset_name: str = "zalando-datasets/fashion_mnist",
) -> grain.DataLoader:
    """Create an infinite, batched Fashion MNIST data loader.

    Images are returned in NHWC format and normalized to ``[-1, 1]`` so they
    can be passed directly to the raw-pixel diffusion model.
    """
    dataset = load_dataset(dataset_name, split=split)
    source = FashionMNISTDataSource(dataset)
    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None,
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        shuffle=shuffle,
        seed=seed,
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=2,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
    )
