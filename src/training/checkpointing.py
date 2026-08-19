"""Unified Orbax and NNX checkpointing utilities for training and inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from orbax.checkpoint import type_handlers

from src.utils.checkpoint import CheckpointManager


def checkpoint_state(state: Any) -> Any:
    """Convert floating-point checkpoint leaves to FP32."""
    return jax.tree.map(
        lambda value: (
            value.astype(jnp.float32)
            if isinstance(value, jax.Array) and jnp.issubdtype(value.dtype, jnp.floating)
            else value
        ),
        state,
    )


def resolve_resume_checkpoint(resume_from: str | Path) -> tuple[Path, int]:
    """Resolve a run directory or checkpoint directory to an Orbax checkpoint root and step."""
    path = Path(resume_from).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {path}")

    if (path / "checkpoints").is_dir():
        checkpoint_root = path / "checkpoints"
        manager = CheckpointManager(str(checkpoint_root))
        step = manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found under {checkpoint_root}")
        return checkpoint_root, int(step)

    if path.name == "checkpoints" and path.is_dir():
        checkpoint_root = path
        manager = CheckpointManager(str(checkpoint_root))
        step = manager.latest_step()
        if step is None:
            raise ValueError(f"No checkpoints found under {checkpoint_root}")
        return checkpoint_root, int(step)

    if path.name.isdigit() and (path / "default").is_dir():
        return path.parent, int(path.name)

    raise ValueError(
        "--resume_from must be a run directory containing checkpoints/ or "
        "an individual Orbax checkpoint directory"
    )


def checkpoint_has_rng(checkpoint_root: Path, step: int) -> bool:
    """Check Orbax metadata for an RNG leaf without deserializing heavy arrays."""
    metadata_path = checkpoint_root / str(step) / "default" / "_METADATA"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return any(key.startswith("('rng'") for key in metadata.get("tree_metadata", {}))
    except Exception:
        return False


def restore_template(
    state: nnx.State,
    sharding: jax.sharding.Sharding | None = None,
) -> Mapping[str, Any]:
    """Build a concrete single-device Orbax restore template from an NNX State."""
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])

    def wrap(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: wrap(child) for key, child in value.items()}
        return {"value": jax.device_put(value, sharding)}

    return wrap(state.to_pure_dict())


def restore_args(
    template: Mapping[str, Any],
    sharding: jax.sharding.Sharding | None = None,
) -> ocp.args.PyTreeRestore:
    """Build array restore arguments for an NNX state template."""
    if sharding is None:
        sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    args = jax.tree.map(lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding), template)
    return ocp.args.PyTreeRestore(template, restore_args=args, partial_restore=True)


def checkpoint_value(state: Mapping[str, Any], path: tuple[Any, ...]) -> Any:
    """Look up a serialized NNX leaf using integer or string path keys."""
    value: Any = state
    for key in path:
        if not isinstance(value, Mapping):
            raise KeyError(path)
        if key in value:
            value = value[key]
        elif str(key) in value:
            value = value[str(key)]
        else:
            raise KeyError(path)
    if isinstance(value, Mapping) and "value" in value:
        value = value["value"]
    return value


def validate_nnx_state(
    target_state: nnx.State,
    checkpoint_state: Mapping[str, Any],
    name: str = "state",
) -> None:
    """Validate serialized paths and shapes before mutating an NNX state."""
    for path, variable in zip(
        target_state.flat_state().paths,
        target_state.flat_state().leaves,
    ):
        try:
            value = checkpoint_value(checkpoint_state, path)
        except KeyError as error:
            raise ValueError(f"Checkpoint {name} is missing state path {path}") from error
        if hasattr(value, "shape") and value.shape != variable.value.shape:
            raise ValueError(
                f"Checkpoint {name} shape mismatch at {path}: "
                f"checkpoint={value.shape}, config={variable.value.shape}"
            )


def restore_nnx_state(
    target_state: nnx.State,
    checkpoint_state: Mapping[str, Any],
    name: str = "state",
) -> None:
    """Assign serialized array values into matching NNX state variables."""
    del name
    for path, variable in zip(
        target_state.flat_state().paths,
        target_state.flat_state().leaves,
    ):
        value = checkpoint_value(checkpoint_state, path)
        if hasattr(value, "dtype") and hasattr(variable.value, "dtype"):
            if jnp.issubdtype(variable.value.dtype, jnp.floating):
                value = value.astype(variable.value.dtype)
        variable.value = value


def restore_training_state(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    ema: Any,
    checkpoint_state: Mapping[str, Any],
    conditioning: str | None = None,
) -> None:
    """Restore model, optimizer, and EMA state after shape and conditioning validation."""
    required = {"model", "ema", "opt"}
    missing = required.difference(checkpoint_state)
    if missing:
        raise ValueError(f"Checkpoint is missing required state groups: {sorted(missing)}")

    if conditioning is not None:
        class_embedding = checkpoint_state["model"].get("y_embedder", {}).get("embedding_table")
        if conditioning == "class" and class_embedding is None:
            raise ValueError("Checkpoint conditioning does not match config: expected class labels")
        if conditioning == "unconditional" and class_embedding is not None:
            raise ValueError(
                "Checkpoint conditioning does not match config: expected unconditional model"
            )

    validate_nnx_state(nnx.state(model), checkpoint_state["model"], "model")
    validate_nnx_state(nnx.state(optimizer), checkpoint_state["opt"], "optimizer")
    validate_nnx_state(ema.ema_state, checkpoint_state["ema"], "EMA")

    restore_nnx_state(nnx.state(model), checkpoint_state["model"], "model")
    restore_nnx_state(nnx.state(optimizer), checkpoint_state["opt"], "optimizer")
    restore_nnx_state(ema.ema_state, checkpoint_state["ema"], "EMA")
