"""Launch the TensorBoard image extraction utility."""

from absl import app

from src.utils.extract_tb_images import main


if __name__ == "__main__":
    app.run(main)
