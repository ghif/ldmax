"""Pytest configuration and global fixtures."""

from absl import flags
import pytest

if not flags.FLAGS.is_parsed():
    flags.FLAGS(["pytest"], known_only=True)


@pytest.fixture(scope="session", autouse=True)
def parse_absl_flags():
    """Ensure absl flags are parsed so grain workers can access them."""
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["pytest"], known_only=True)
