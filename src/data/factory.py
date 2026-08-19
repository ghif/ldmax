"""Unified dataset factory for training and validation loaders."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import jax

from src.data.celeba import get_celeba_dataset
from src.data.cifar import get_cifar10_dataset
from src.data.fashion_mnist import get_fashion_mnist_dataset
from src.utils.prefetch import DevicePrefetcher


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get configuration attribute from Dict, ConfigDict, or Namespace."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if hasattr(obj, "get") and callable(obj.get):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing dataset format, resolution, and conditioning."""

    name: str
    is_latent: bool
    image_size: int
    channels: int
    num_classes: int
    label_mode: str
    label_dim: int | None = None


@dataclass
class DataLoaderBundle:
    """Container holding training and validation data iterators and metadata."""

    train_iter: Iterator[dict[str, Any]]
    val_iter: Iterator[dict[str, Any]] | None
    metadata: DatasetMetadata


def get_dataset_metadata(config: Any) -> DatasetMetadata:
    """Extract standard metadata from configuration."""
    dataset_name = _cfg_get(config, "dataset", "cifar10").lower()
    model_cfg = config.model
    in_channels = model_cfg.in_channels
    is_latent = in_channels == 4 or dataset_name == "celeba"

    label_mode = _cfg_get(model_cfg, "label_mode", None)
    if label_mode is None:
        conditioning = _cfg_get(model_cfg, "conditioning", "class")
        label_mode = "class" if conditioning == "class" else "none"

    label_dim = _cfg_get(model_cfg, "label_dim", None)
    if label_mode == "attributes" and label_dim is None:
        label_dim = 40

    data_cfg = getattr(config, "data", None)
    if dataset_name == "celeba":
        image_size = _cfg_get(data_cfg, "image_size", 256) if data_cfg is not None else 256
        channels = 3  # Raw RGB images before VAE encoding
        num_classes = _cfg_get(model_cfg, "num_classes", 40)
    elif dataset_name in {"fashion_mnist", "fashion-mnist"}:
        image_size = model_cfg.input_size
        channels = 1
        num_classes = _cfg_get(model_cfg, "num_classes", 10)
    else:  # CIFAR-10 / default
        image_size = model_cfg.input_size
        channels = 3
        num_classes = _cfg_get(model_cfg, "num_classes", 10)

    return DatasetMetadata(
        name=dataset_name,
        is_latent=is_latent,
        image_size=image_size,
        channels=channels,
        num_classes=num_classes,
        label_mode=label_mode,
        label_dim=label_dim,
    )


def create_dataloaders(
    config: Any,
    sharding: jax.sharding.Sharding | None = None,
) -> DataLoaderBundle:
    """Create train and validation data loaders according to config.

    Args:
        config: Experiment configuration object.
        sharding: Optional device sharding for prefetching.

    Returns:
        DataLoaderBundle containing train_iter, val_iter, and metadata.
    """
    metadata = get_dataset_metadata(config)
    dataset_name = metadata.name
    train_cfg = config.training
    batch_size = train_cfg.batch_size
    seed = train_cfg.seed
    prefetch_size = _cfg_get(train_cfg, "prefetch_size", 0)

    data_cfg = getattr(config, "data", None)

    if dataset_name in {"cifar10", "cifar"}:
        target_size = _cfg_get(data_cfg, "image_size", None) if data_cfg is not None else None
        train_data = get_cifar10_dataset(
            batch_size=batch_size,
            split="train",
            shuffle=True,
            seed=seed,
            target_size=target_size,
        )
        val_data = get_cifar10_dataset(
            batch_size=batch_size,
            split="test",
            shuffle=False,
            seed=seed,
            target_size=target_size,
        )
    elif dataset_name in {"fashion_mnist", "fashion-mnist"}:
        custom_path = _cfg_get(data_cfg, "dataset_name", None) if data_cfg is not None else None
        kwargs = {"dataset_name": custom_path} if custom_path else {}
        train_data = get_fashion_mnist_dataset(
            batch_size=batch_size,
            split="train",
            shuffle=True,
            seed=seed,
            **kwargs,
        )
        val_data = get_fashion_mnist_dataset(
            batch_size=batch_size,
            split="test",
            shuffle=False,
            seed=seed,
            **kwargs,
        )
    elif dataset_name == "celeba":
        image_size = _cfg_get(data_cfg, "image_size", 256) if data_cfg is not None else 256
        custom_path = _cfg_get(data_cfg, "dataset_name", None) if data_cfg is not None else None
        kwargs = {"dataset_name": custom_path} if custom_path else {}
        train_data = get_celeba_dataset(
            batch_size=batch_size,
            split="train",
            shuffle=True,
            seed=seed,
            target_size=image_size,
            **kwargs,
        )
        val_data = get_celeba_dataset(
            batch_size=batch_size,
            split="val",
            shuffle=False,
            seed=seed,
            target_size=image_size,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name!r}")

    if prefetch_size > 0:
        if sharding is None:
            sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        train_iter = iter(DevicePrefetcher(train_data, sharding, prefetch_size))
        val_iter = iter(DevicePrefetcher(val_data, sharding, 1)) if val_data is not None else None
    else:
        train_iter = iter(train_data)
        val_iter = iter(val_data) if val_data is not None else None

    return DataLoaderBundle(
        train_iter=train_iter,
        val_iter=val_iter,
        metadata=metadata,
    )
