"""Utilities for overlapping host-to-device transfers with computation."""

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterable

import jax


class DevicePrefetcher:
    """A wrapper around an iterator that prefetches data to devices.

    This helps hide the latency of host-to-device transfers (jax.device_put)
    by running them in the background while the TPU is computing.
    """

    def __init__(
        self,
        iterable: Iterable[Any],
        sharding: jax.sharding.Sharding,
        prefetch_size: int = 2,
    ):
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
        """Yield batches after staging them on the target device."""
        iterator = iter(self.iterable)
        queue: deque[Future] = deque()
        executor = ThreadPoolExecutor(max_workers=1)

        def get_next_sharded():
            try:
                batch = next(iterator)
                # Shard the batch (recursively if it's a dict)
                return jax.tree.map(lambda x: jax.device_put(x, self.sharding), batch)
            except StopIteration:
                return None

        try:
            # The worker performs both host batch preparation and device transfer.
            for _ in range(self.prefetch_size):
                queue.append(executor.submit(get_next_sharded))

            while queue:
                item = queue.popleft().result()
                if item is None:
                    break
                yield item
                queue.append(executor.submit(get_next_sharded))
        finally:
            executor.shutdown(wait=True)
