"""Launch raw-pixel Fashion MNIST diffusion training."""

from absl import app, flags

from src.training.trainer import Trainer

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/fashion_mnist.yaml", "Path to YAML configuration file.")
flags.DEFINE_string("output_dir", "", "Directory for logs and checkpoints.")
flags.DEFINE_string("resume_from", "", "Optional run or checkpoint directory to resume from.")


def main(_):
    """Main CLI entry point for Fashion MNIST training."""
    trainer = Trainer(
        config=FLAGS.config,
        output_dir=FLAGS.output_dir,
        resume_from=FLAGS.resume_from,
    )
    trainer.run()


if __name__ == "__main__":
    app.run(main)
