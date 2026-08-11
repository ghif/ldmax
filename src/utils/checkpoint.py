"""Orbax checkpointing utilities."""

import hashlib
import os
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import orbax.checkpoint as ocp
from google.cloud import storage


def materialize_checkpoint(checkpoint: str) -> Path:
    """Return a local checkpoint path, downloading a GCS checkpoint if needed.

    GCS checkpoints are stored as directory trees by Orbax.  Sampling needs
    the complete tree locally because Orbax restores manifests and array data
    from several files beneath the checkpoint prefix.
    """
    if not checkpoint.startswith("gs://"):
        return Path(checkpoint).expanduser().resolve()

    parsed = urlparse(checkpoint)
    prefix = parsed.path.strip("/")
    step_name = Path(prefix).name
    if not parsed.netloc or not prefix or not step_name.isdigit():
        raise ValueError(
            "A GCS checkpoint must point to an individual numeric step, "
            "for example gs://bucket/models/run/checkpoints/12000"
        )

    cache_root = Path(
        os.environ.get("LDMAX_CHECKPOINT_CACHE", "~/.cache/ldmax/checkpoints")
    ).expanduser()
    cache_key = hashlib.sha256(checkpoint.encode("utf-8")).hexdigest()[:16]
    local_checkpoint = cache_root / cache_key / "checkpoints" / step_name
    complete_marker = local_checkpoint / ".download_complete"
    if complete_marker.is_file():
        return local_checkpoint

    client = storage.Client()
    blobs = list(client.list_blobs(parsed.netloc, prefix=prefix.rstrip("/") + "/"))
    if not blobs:
        raise FileNotFoundError(f"No GCS checkpoint files found under {checkpoint}")

    download_root = local_checkpoint.parent.parent / f".{step_name}.partial"
    if download_root.exists():
        for path in sorted(download_root.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    local_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    prefix_with_slash = prefix.rstrip("/") + "/"
    for blob in blobs:
        relative_path = blob.name[len(prefix_with_slash) :]
        if not relative_path:
            continue
        destination = download_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))

    if local_checkpoint.exists():
        for path in sorted(local_checkpoint.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        local_checkpoint.rmdir()
    download_root.rename(local_checkpoint)
    complete_marker.touch()
    return local_checkpoint


class CheckpointManager:
    """Manages local Orbax checkpoints and optional GCS synchronization."""

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 5,
        gcs_directory: str | None = None,
        artifact_paths: list[str] | None = None,
        best_metric: str | None = None,
        best_mode: str = "min",
    ):
        """Initialize the checkpoint manager.

        Args:
            directory: Directory to save checkpoints in.
            max_to_keep: Maximum number of recent checkpoints to keep.
            gcs_directory: Optional GCS destination for synchronized artifacts.
            artifact_paths: Additional local files or directories to synchronize.
            best_metric: Optional metric name used to retain the best checkpoint.
            best_mode: Whether a lower or higher best metric is preferred.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gcs_directory = gcs_directory
        self.artifact_paths = [Path(path) for path in artifact_paths or []]
        self.best_metric = best_metric
        if best_mode not in {"min", "max"}:
            raise ValueError("best_mode must be 'min' or 'max'")

        # Initialize checkpointer
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            best_fn=(lambda metrics: metrics[best_metric]) if best_metric else None,
            best_mode=best_mode,
            create=True
        )

        # Initialize checkpointer using modern API
        self.manager = ocp.CheckpointManager(
            self.directory.absolute(),
            ocp.PyTreeCheckpointer(),
            options=options
        )

    def save(
        self,
        step: int,
        state: Mapping[str, Any],
        metrics: Mapping[str, Any] | None = None,
    ):
        """Save the current training state.

        Args:
            step: Current training step.
            state: A PyTree of the state to save.
            metrics: Optional metrics used for best-checkpoint retention.
        """
        if self.best_metric is not None:
            if metrics is None or self.best_metric not in metrics:
                raise ValueError(
                    f"Checkpoint step {step} requires metric '{self.best_metric}'"
                )
        self.manager.save(step, state, metrics=metrics)
        if self.gcs_directory is not None:
            self.sync_to_gcs()

    def sync_to_gcs(self) -> None:
        """Upload the complete local checkpoint tree to the configured GCS path."""
        if self.gcs_directory is None:
            return

        parsed = urlparse(self.gcs_directory)
        if parsed.scheme != "gs" or not parsed.netloc:
            raise ValueError("gcs_directory must be a GCS URL such as 'gs://diffjax/models'")

        run_root = self.directory.parent
        run_name = run_root.name
        remote_prefix = "/".join(part for part in [parsed.path.strip("/"), run_name] if part)
        bucket = storage.Client().bucket(parsed.netloc)

        paths_to_sync = [self.directory, *self.artifact_paths]
        for root in paths_to_sync:
            if root.is_file():
                files = [root]
            elif root.is_dir():
                files = [path for path in sorted(root.rglob("*")) if path.is_file()]
            else:
                continue

            for path in files:
                relative_path = path.relative_to(run_root).as_posix()
                bucket.blob(f"{remote_prefix}/{relative_path}").upload_from_filename(str(path))

    def restore(
        self,
        step: int = None,
        items: Any = None,
        partial_restore: bool = True,
        restore_kwargs: Mapping[str, Any] | None = None,
        args: Any = None,
    ) -> Any:
        """Restore state from a checkpoint.

        Args:
            step: Specific step to restore. If None, restores latest.
            items: A template (e.g. state dictionary) to guide restoration.
            partial_restore: Whether to allow partial restoration if structures mismatch.
            restore_kwargs: Additional Orbax restore arguments.
            args: Explicit Orbax restore arguments, when required.

        Returns:
            The restored PyTree.
        """
        if step is None:
            step = self.manager.latest_step()

        if step is None:
            return None

        if args is not None:
            return self.manager.restore(step, args=args)
        kwargs = dict(restore_kwargs or {})
        kwargs.setdefault("partial_restore", partial_restore)
        return self.manager.restore(step, items=items, restore_kwargs=kwargs)

    def latest_step(self) -> int:
        """Get the latest checkpoint step."""
        return self.manager.latest_step()
