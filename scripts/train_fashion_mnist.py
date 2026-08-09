"""Launch raw-pixel Fashion MNIST diffusion training."""

from absl import app

from src.training.fashion_mnist_runner import main


if __name__ == "__main__":
    app.run(main)
