"""Unified model factory for Diffusion Transformer (DiT) architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from src.models.dit.dit import DiT


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get configuration attribute from Dict, ConfigDict, or Namespace."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    if hasattr(obj, "get") and callable(obj.get):
        return obj.get(key, default)
    return getattr(obj, key, default)


def create_model(config: Any, rng_key: jax.Array) -> nnx.Module:
    """Instantiate a DiT diffusion backbone from configuration.

    Args:
        config: Experiment configuration object.
        rng_key: PRNGKey for parameter initialization.

    Returns:
        An instantiated Flax NNX Module.
    """
    model_cfg = config.model
    train_cfg = getattr(config, "training", None)

    use_bf16 = False
    if train_cfg is not None:
        use_bf16 = bool(
            _cfg_get(train_cfg, "use_bf16", False) or _cfg_get(train_cfg, "mixed_precision", False)
        )

    compute_dtype = jnp.bfloat16 if use_bf16 else None

    # Resolve conditioning and label mode
    label_mode = _cfg_get(model_cfg, "label_mode", None)
    if label_mode is None:
        conditioning = _cfg_get(model_cfg, "conditioning", "class")
        label_mode = "class" if conditioning == "class" else "none"

    label_dim = _cfg_get(model_cfg, "label_dim", None)
    if label_mode == "attributes" and label_dim is None:
        label_dim = 40

    model_type = _cfg_get(model_cfg, "type", "dit").lower()

    if model_type == "dit":
        return DiT(
            input_size=model_cfg.input_size,
            patch_size=model_cfg.patch_size,
            in_channels=model_cfg.in_channels,
            hidden_size=model_cfg.hidden_size,
            depth=model_cfg.depth,
            num_heads=model_cfg.num_heads,
            mlp_ratio=_cfg_get(model_cfg, "mlp_ratio", 4.0),
            num_classes=_cfg_get(model_cfg, "num_classes", 10),
            label_mode=label_mode,
            label_dim=label_dim,
            label_dropout_prob=_cfg_get(model_cfg, "label_dropout_prob", 0.1),
            learn_sigma=_cfg_get(model_cfg, "learn_sigma", False),
            compute_dtype=compute_dtype,
            rngs=nnx.Rngs(rng_key),
        )

    raise ValueError(f"Unsupported model type: {model_type!r}. Supported types: 'dit'.")
