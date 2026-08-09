"""Launch the LDMAX sampling application."""

from absl import app

from src.sampling.generate import main


if __name__ == "__main__":
    app.run(main)
