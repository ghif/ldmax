"""Orbax checkpointing utilities."""

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import orbax.checkpoint as ocp
from google.cloud import storage


class CheckpointManager:
    """Manages local Orbax checkpoints and optional GCS synchronization."""

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 5,
        gcs_directory: str | None = None,
        artifact_paths: list[str] | None = None,
    ):
        """Initialize the checkpoint manager.

        Args:
            directory: Directory to save checkpoints in.
            max_to_keep: Maximum number of recent checkpoints to keep.
            gcs_directory: Optional GCS destination for synchronized artifacts.
            artifact_paths: Additional local files or directories to synchronize.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gcs_directory = gcs_directory
        self.artifact_paths = [Path(path) for path in artifact_paths or []]

        # Initialize checkpointer
        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True
        )

        # Initialize checkpointer using modern API
        self.manager = ocp.CheckpointManager(
            self.directory.absolute(),
            ocp.PyTreeCheckpointer(),
            options=options
        )

    def save(self, step: int, state: Mapping[str, Any]):
        """Save the current training state.

        Args:
            step: Current training step.
            state: A PyTree of the state to save.
        """
        self.manager.save(step, state)
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
