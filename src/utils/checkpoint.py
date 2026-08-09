"""Orbax checkpointing utilities."""

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from google.cloud import storage
import orbax.checkpoint as ocp


class CheckpointManager:
    """Manages local Orbax checkpoints and optional GCS synchronization."""

    def __init__(self, directory: str, max_to_keep: int = 5, gcs_directory: str | None = None):
        """Initialize the checkpoint manager.

        Args:
            directory: Directory to save checkpoints in.
            max_to_keep: Maximum number of recent checkpoints to keep.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.gcs_directory = gcs_directory
        
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

        run_name = self.directory.parent.name
        checkpoint_name = self.directory.name
        prefix_parts = [parsed.path.strip("/"), run_name, checkpoint_name]
        remote_prefix = "/".join(part for part in prefix_parts if part)
        bucket = storage.Client().bucket(parsed.netloc)

        for path in sorted(self.directory.rglob("*")):
            if path.is_file():
                relative_path = path.relative_to(self.directory).as_posix()
                bucket.blob(f"{remote_prefix}/{relative_path}").upload_from_filename(str(path))

    def restore(self, step: int = None, items: Any = None, partial_restore: bool = True) -> Any:
        """Restore state from a checkpoint.

        Args:
            step: Specific step to restore. If None, restores latest.
            items: A template (e.g. state dictionary) to guide restoration.
            partial_restore: Whether to allow partial restoration if structures mismatch.

        Returns:
            The restored PyTree.
        """
        if step is None:
            step = self.manager.latest_step()
        
        if step is None:
            return None
            
        return self.manager.restore(step, items=items, restore_kwargs={'partial_restore': partial_restore})

    def latest_step(self) -> int:
        """Get the latest checkpoint step."""
        return self.manager.latest_step()
