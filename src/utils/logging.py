"""TensorBoard logging utilities."""

import os
from typing import Any, Mapping
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    """Wrapper for TensorBoard SummaryWriter."""

    def __init__(self, log_dir: str):
        """Initialize the logger.

        Args:
            log_dir: Directory to save logs in.
        """
        self.writer = SummaryWriter(log_dir)

    def log_scalars(self, step: int, metrics: Mapping[str, float]):
        """Log multiple scalar values.

        Args:
            step: Current step.
            metrics: Dictionary of metric names and values.
        """
        import numpy as np
        for name, value in metrics.items():
            if hasattr(value, "tolist"):
                value = float(value)
            self.writer.add_scalar(name, value, step)

    def log_images(self, step: int, name: str, images: Any):
        """Log image grids.

        Args:
            step: Current step.
            name: Label for the images.
            images: Image data (numpy array or torch tensor).
        """
        import numpy as np
        if hasattr(images, "device"): # JAX array or torch tensor
            images = np.array(images)
        self.writer.add_images(name, images, step, dataformats="NHWC")

    def close(self):
        """Close the writer."""
        self.writer.close()
