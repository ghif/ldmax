"""Launch native-pixel CIFAR10 inference."""

from absl import app

from src.sampling.cifar10 import main


if __name__ == "__main__":
    app.run(main)
