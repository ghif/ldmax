"""Launch the LDMAX training application."""

from absl import app

from src.training.runner import main


if __name__ == "__main__":
    app.run(main)
