"""EMA weight tracking utilities."""

import jax
from flax import nnx
from typing import Any

class EMAManager:
    """Manages Exponential Moving Average of model parameters."""

    def __init__(self, model: nnx.Module, decay: float = 0.999):
        """Initialize EMA.

        Args:
            model: The model to track.
            decay: EMA decay rate.
        """
        self.decay = decay
        # Get a snapshot of the model state for EMA
        self.ema_state = nnx.state(model)

    def update(self, model: nnx.Module):
        """Update EMA weights.
        
        Args:
            model: The current model with updated weights.
        """
        current_state = nnx.state(model)
        
        # Simple EMA update: ema = decay * ema + (1 - decay) * current
        self.ema_state = jax.tree.map(
            lambda e, c: self.decay * e + (1 - self.decay) * c if isinstance(c, jax.Array) else c,
            self.ema_state,
            current_state
        )

    def apply_to(self, model: nnx.Module):
        """Apply EMA weights to a model instance."""
        nnx.update(model, self.ema_state)

    @property
    def state(self) -> nnx.State:
        """Get the EMA state."""
        return self.ema_state
