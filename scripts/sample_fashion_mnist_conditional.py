"""Launch class-conditional raw-pixel Fashion MNIST inference."""

from absl import app

from src.sampling.fashion_mnist import main


if __name__ == "__main__":
    app.run(main)
