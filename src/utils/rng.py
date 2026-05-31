"""Centralized PRNGKey management for reproducibility."""

import jax
import jax.numpy as jnp

class RNGManager:
    """Manages PRNGKeys for JAX operations."""

    def __init__(self, seed: int):
        """Initialize with a base seed.

        Args:
            seed: The initial integer seed.
        """
        self._key = jax.random.PRNGKey(seed)

    def next(self) -> jax.Array:
        """Get the next PRNGKey.

        Returns:
            A new PRNGKey.
        """
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def split(self, num: int = 2) -> jax.Array:
        """Split the current key into multiple keys.

        Args:
            num: Number of keys to split into.

        Returns:
            An array of PRNGKeys.
        """
        keys = jax.random.split(self._key, num + 1)
        self._key = keys[0]
        return keys[1:]
