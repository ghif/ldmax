"""Grain-based data pipeline for CelebA with 40 binary attribute labels from GCS."""

import io
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import grain.python as grain
import numpy as np
import pandas as pd
from google.cloud import storage
from PIL import Image

DEFAULT_DATASET_PATH = "gs://diffjax/datasets/celeba"

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

PARTITION_MAP = {
    "train": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "test": 2,
}


def _read_bytes(path_or_url: str, client: storage.Client | None = None) -> bytes:
    """Read bytes from a GCS URI or local filesystem path."""
    parsed = urlparse(path_or_url)
    if parsed.scheme == "gs":
        bucket_name = parsed.netloc
        blob_path = parsed.path.lstrip("/")
        if client is None:
            client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_bytes()
    return Path(path_or_url).read_bytes()


class CelebADataSource(grain.RandomAccessDataSource):
    """A Grain-compatible data source reading CelebA images on-the-fly from GCS."""

    CELEBA_ATTRIBUTE_NAMES = CELEBA_ATTRIBUTE_NAMES

    def __init__(
        self,
        dataset_path: str = DEFAULT_DATASET_PATH,
        split: str = "train",
        target_size: int = 64,
    ):
        """Initialize the CelebA data source from GCS or local directory.

        Args:
            dataset_path: GCS directory (e.g. 'gs://diffjax/datasets/celeba') or local path.
            split: Dataset split ('train', 'val'/'valid'/'validation', 'test', or 'all').
            target_size: Target square image resolution after center crop.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.target_size = target_size

        parsed = urlparse(dataset_path)
        self.is_gcs = parsed.scheme == "gs"
        if self.is_gcs:
            if not parsed.netloc or not parsed.path.strip("/"):
                raise ValueError(
                    f"dataset_path must be a valid GCS directory such as {DEFAULT_DATASET_PATH!r}"
                )
            self.bucket_name = parsed.netloc
            self.prefix = parsed.path.strip("/")
            client = storage.Client()
            part_bytes = _read_bytes(
                f"gs://{self.bucket_name}/{self.prefix}/list_eval_partition.csv", client
            )
            attr_bytes = _read_bytes(
                f"gs://{self.bucket_name}/{self.prefix}/list_attr_celeba.csv", client
            )
        else:
            local_root = Path(dataset_path).expanduser().resolve()
            if not local_root.exists():
                raise FileNotFoundError(f"CelebA dataset directory does not exist: {local_root}")
            self.bucket_name = None
            self.prefix = str(local_root)
            part_bytes = (local_root / "list_eval_partition.csv").read_bytes()
            attr_bytes = (local_root / "list_attr_celeba.csv").read_bytes()

        df_part = pd.read_csv(io.BytesIO(part_bytes))
        df_attr = pd.read_csv(io.BytesIO(attr_bytes))
        df_merged = pd.merge(df_part, df_attr, on="image_id")

        if split in PARTITION_MAP:
            target_partition = PARTITION_MAP[split]
            split_df = df_merged[df_merged["partition"] == target_partition]
        elif split == "all":
            split_df = df_merged
        else:
            raise ValueError(
                f"Unsupported CelebA split: {split!r}. Expected one of {list(PARTITION_MAP.keys()) + ['all']}"
            )

        if len(split_df) == 0:
            raise ValueError(f"No records found for CelebA split {split!r} in {dataset_path}")

        self.image_ids = split_df["image_id"].tolist()
        raw_labels = split_df[list(self.CELEBA_ATTRIBUTE_NAMES)].to_numpy(dtype=np.int32)
        # Convert CelebA -1 / 1 attribute encoding to 0 / 1 binary indicators
        self.labels = np.where(raw_labels > 0, 1, 0).astype(np.int32)

        # Lazy storage client for multi-worker grain DataLoader processes
        self._client: storage.Client | None = None
        self._bucket: Any = None

    @property
    def bucket(self) -> Any:
        """Lazily initialize and return the GCS bucket instance."""
        if self._bucket is None:
            self._client = storage.Client()
            self._bucket = self._client.bucket(self.bucket_name)
        return self._bucket

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.image_ids)

    def _fetch_image_bytes(self, image_id: str) -> bytes:
        """Fetch image bytes directly from GCS or local filesystem."""
        if self.is_gcs:
            blob = self.bucket.blob(f"{self.prefix}/img_align_celeba/img_align_celeba/{image_id}")
            return blob.download_as_bytes()
        local_path = Path(self.prefix) / "img_align_celeba" / "img_align_celeba" / image_id
        if not local_path.exists():
            local_path = Path(self.prefix) / "img_align_celeba" / image_id
        if not local_path.exists():
            local_path = Path(self.prefix) / image_id
        return local_path.read_bytes()

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        """Fetch and preprocess a single CelebA sample."""
        image_id = self.image_ids[idx]
        img_bytes = self._fetch_image_bytes(image_id)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Center crop to square
        w, h = img.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        right = (w + crop_size) // 2
        bottom = (h + crop_size) // 2
        img = img.crop((left, top, right, bottom))

        # Resize to target resolution if specified
        if self.target_size is not None and (
            img.size[0] != self.target_size or img.size[1] != self.target_size
        ):
            img = img.resize((self.target_size, self.target_size), Image.LANCZOS)

        # Normalize [0, 255] -> [-1.0, 1.0]
        img_np = np.asarray(img, dtype=np.float32)
        img_normalized = (img_np / 127.5) - 1.0

        return {
            "image": img_normalized,
            "label": self.labels[idx],
        }


def get_celeba_dataset(
    batch_size: int,
    split: str = "train",
    shuffle: bool = True,
    seed: int = 0,
    target_size: int = 64,
    dataset_name: str = DEFAULT_DATASET_PATH,
    dataset_config: str | None = None,
    worker_count: int = 4,
) -> grain.DataLoader:
    """Initialize the CelebA data loader using Grain reading directly from GCS.

    Args:
        batch_size: Number of elements per batch.
        split: Dataset split ('train', 'val'/'valid'/'validation', 'test', or 'all').
        shuffle: Whether to shuffle the data.
        seed: Random seed for shuffling.
        target_size: The resolution to resize images to.
        dataset_name: GCS directory (e.g. 'gs://diffjax/datasets/celeba') or local path.
        dataset_config: Deprecated / unused argument kept for API compatibility.
        worker_count: Number of worker processes for data loading.

    Returns:
        A Grain DataLoader instance.
    """
    del dataset_config
    source = CelebADataSource(
        dataset_path=dataset_name,
        split=split,
        target_size=target_size,
    )

    transformations = [grain.Batch(batch_size=batch_size, drop_remainder=True)]

    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None,
        shard_options=grain.ShardOptions(shard_index=0, shard_count=1),
        shuffle=shuffle,
        seed=seed,
    )

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=worker_count,
        operations=transformations,
    )

    return loader
