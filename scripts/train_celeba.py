"""Launch CelebA Latent VAE Diffusion Transformer training."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from absl import app

from src.training.celeba_runner import main

if __name__ == "__main__":
    app.run(main)
