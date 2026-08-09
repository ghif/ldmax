"""Launch the CelebA attribute-grid sampler."""

from absl import app

from src.sampling.celeba_grid import main


if __name__ == "__main__":
    app.run(main)
