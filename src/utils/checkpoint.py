"""Orbax checkpointing utilities."""

import os
from pathlib import Path
from typing import Any, Mapping

import orbax.checkpoint as ocp

class CheckpointManager:
    """Manages training checkpoints using Orbax."""

    def __init__(self, directory: str, max_to_keep: int = 5):
        """Initialize the checkpoint manager.

        Args:
            directory: Directory to save checkpoints in.
            max_to_keep: Maximum number of recent checkpoints to keep.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        
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

    def restore(self, step: int = None) -> Any:
        """Restore state from a checkpoint.

        Args:
            step: Specific step to restore. If None, restores latest.

        Returns:
            The restored PyTree.
        """
        if step is None:
            step = self.manager.latest_step()
        
        if step is None:
            return None
            
        return self.manager.restore(step)

    def latest_step(self) -> int:
        """Get the latest checkpoint step."""
        return self.manager.latest_step()
