"""Launch native-pixel CIFAR10 diffusion training."""

from absl import app

from src.training.cifar10_runner import main

if __name__ == "__main__":
    app.run(main)
