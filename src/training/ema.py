"""EMA weight tracking utilities."""

import jax
import jax.numpy as jnp
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

    @staticmethod
    @jax.jit
    def _ema_update(ema_state, current_state, decay):
        def update_fn(e, c):
            # Only update floating point parameters, skip keys or ints
            if isinstance(c, jax.Array) and jnp.issubdtype(c.dtype, jnp.floating):
                return decay * e + (1.0 - decay) * c
            return c
        return jax.tree.map(update_fn, ema_state, current_state)

    def update(self, model: nnx.Module):
        """Update EMA weights.
        
        Args:
            model: The current model with updated weights.
        """
        current_state = nnx.state(model)
        self.ema_state = self._ema_update(self.ema_state, current_state, self.decay)

    def apply_to(self, model: nnx.Module):
        """Apply EMA weights to a model instance."""
        nnx.update(model, self.ema_state)

    @property
    def state(self) -> nnx.State:
        """Get the EMA state."""
        return self.ema_state
