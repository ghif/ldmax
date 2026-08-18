import sys
from pathlib import Path

from absl import flags
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not flags.FLAGS.is_parsed():
    flags.FLAGS(["pytest"], known_only=True)


@pytest.fixture(scope="session", autouse=True)
def parse_absl_flags():
    """Ensure absl flags are parsed so grain workers can access them."""
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["pytest"], known_only=True)
