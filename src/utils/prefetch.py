"""Utilities for overlapping host-to-device transfers with computation."""

import jax
from typing import Iterable, Any

class DevicePrefetcher:
    """A wrapper around an iterator that prefetches data to devices.
    
    This helps hide the latency of host-to-device transfers (jax.device_put)
    by running them in the background while the TPU is computing.
    """

    def __init__(self, iterable: Iterable[Any], sharding: jax.sharding.Sharding, prefetch_size: int = 2):
        """Initialize the prefetcher.

        Args:
            iterable: The data iterator (e.g., Grain DataLoader).
            sharding: The sharding to apply to the data.
            prefetch_size: Number of batches to prefetch.
        """
        self.iterable = iterable
        self.sharding = sharding
        self.prefetch_size = prefetch_size

    def __iter__(self):
        iterator = iter(self.iterable)
        queue = []

        def get_next_sharded():
            try:
                batch = next(iterator)
                # Shard the batch (recursively if it's a dict)
                return jax.tree.map(lambda x: jax.device_put(x, self.sharding), batch)
            except StopIteration:
                return None

        # Initial prefetch
        for _ in range(self.prefetch_size):
            item = get_next_sharded()
            if item is not None:
                queue.append(item)

        while queue:
            yield queue.pop(0)
            item = get_next_sharded()
            if item is not None:
                queue.append(item)
