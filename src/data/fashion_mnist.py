"""Simple NumPy data iterator for raw-pixel Fashion MNIST training."""

from collections.abc import Iterator
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse

import numpy as np
from datasets import Dataset
from google.cloud import storage


DEFAULT_DATASET_PATH = "gs://diffjax/datasets/fashion-mnist/huggingface"


def _load_dataset_from_gcs(dataset_path: str, split: str) -> Dataset:
    """Load one Fashion MNIST Arrow split from a GCS dataset directory."""
    parsed = urlparse(dataset_path)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(
            "dataset_path must be a GCS directory such as "
            f"{DEFAULT_DATASET_PATH!r}"
        )
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported Fashion MNIST split: {split!r}")

    bucket_name = parsed.netloc
    prefix = parsed.path.strip("/")
    blob_name = f"{prefix}/fashion_mnist-{split}.arrow"
    cache_dir = Path.home() / ".cache" / "ldmax" / "fashion_mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"fashion_mnist-{split}.arrow"

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.reload()
    if cache_path.exists() and cache_path.stat().st_size == blob.size:
        return Dataset.from_file(str(cache_path))

    with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".arrow", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        blob.download_to_filename(str(temporary_path))
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return Dataset.from_file(str(cache_path))


def get_fashion_mnist_dataset(
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 0,
    dataset_name: str = DEFAULT_DATASET_PATH,
) -> Iterator[dict[str, np.ndarray]]:
    """Create an infinite, fixed-size Fashion MNIST batch iterator.

    Images are returned in NHWC format and normalized to ``[-1, 1]`` so they
    can be passed directly to the raw-pixel diffusion model.

    The Arrow split is read from the configured GCS directory and cached
    locally for reuse. The iterator reshuffles the training indices at the
    beginning of every epoch and drops the final partial batch so every yielded
    batch has a static shape.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    dataset = _load_dataset_from_gcs(dataset_name, split)
    if len(dataset) < batch_size:
        raise ValueError(
            f"batch_size ({batch_size}) cannot exceed dataset size ({len(dataset)})"
        )

    indices = np.arange(len(dataset), dtype=np.int64)
    random = np.random.default_rng(seed)

    while True:
        if shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_indices = indices[start : start + batch_size]
            images = []
            labels = []
            for index in batch_indices:
                item: Any = dataset[int(index)]
                image = np.asarray(item["image"], dtype=np.float32) / 127.5 - 1.0
                images.append(image[..., None])  # NHWC: 28 x 28 x 1
                labels.append(np.int32(item["label"]))

            yield {
                "image": np.stack(images, axis=0),
                "label": np.asarray(labels, dtype=np.int32),
            }
