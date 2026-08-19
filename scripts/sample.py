"""Launch LDMAX standalone diffusion sampling."""

from absl import app, flags

from src.sampling.generator import generate_samples

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/cifar10_pixel.yaml", "Path to YAML configuration file.")
flags.DEFINE_string("checkpoint", "", "Optional path to checkpoint.")
flags.DEFINE_integer("num_samples", 16, "Number of images to generate.")
flags.DEFINE_integer("num_inference_steps", 50, "Number of DDIM sampling steps.")
flags.DEFINE_float("cfg_scale", 1.5, "Classifier-Free Guidance scale.")
flags.DEFINE_integer("class_id", -1, "Class ID to sample, or -1 for diverse class grid.")
flags.DEFINE_string("attribute_names", "Smiling", "Comma-separated CelebA attribute names.")
flags.DEFINE_integer("seed", 42, "Random seed for sampling noise.")
flags.DEFINE_string("output_path", "./samples.png", "Path for the output image grid.")
flags.DEFINE_bool("use_ema", True, "Whether to sample using EMA weights from checkpoint.")


def main(_):
    """Main CLI entry point for standalone sampling."""
    generate_samples(
        config=FLAGS.config,
        checkpoint=FLAGS.checkpoint,
        num_samples=FLAGS.num_samples,
        num_inference_steps=FLAGS.num_inference_steps,
        cfg_scale=FLAGS.cfg_scale,
        class_id=FLAGS.class_id,
        attribute_names=FLAGS.attribute_names,
        seed=FLAGS.seed,
        output_path=FLAGS.output_path,
        use_ema=FLAGS.use_ema,
    )


if __name__ == "__main__":
    app.run(main)
