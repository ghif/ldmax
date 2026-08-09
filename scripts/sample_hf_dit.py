"""Launch the Hugging Face DiT sampler."""

from absl import app

from src.sampling.hf_dit import main


if __name__ == "__main__":
    app.run(main)
