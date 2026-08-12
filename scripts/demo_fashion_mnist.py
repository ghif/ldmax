"""Interactive Gradio demo for class-conditional Fashion MNIST sampling."""

import argparse
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

try:
    import gradio as gr
except ImportError as error:  # pragma: no cover - exercised by the CLI
    raise SystemExit(
        "Gradio is required for this demo. Install the project dependencies "
        "or run: python -m pip install gradio"
    ) from error

from flax import nnx

from src.models.dit.dit import DiT, resolve_conditioning_mode
from src.sampling.fashion_mnist import _restore_ema
from src.training.sampler import DDIMSampler
from src.utils.config import load_config
from src.utils.rng import RNGManager


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

CLASS_LEGEND = " | ".join(
    f"**{index}** {name}" for index, name in enumerate(CLASS_NAMES)
)


def _build_demo_model(config: Any, seed: int) -> DiT:
    """Build the checkpoint-compatible model for the active JAX backend."""
    conditioning = config.model.get("conditioning", "class")
    if conditioning != "class":
        raise ValueError("This demo requires a class-conditioned Fashion MNIST model")
    use_bf16 = config.training.get("use_bf16", False)
    use_bf16 = use_bf16 and jax.devices()[0].platform == "tpu"
    config.training.use_bf16 = use_bf16
    return DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_classes=config.model.num_classes,
        label_mode=resolve_conditioning_mode(conditioning),
        label_dropout_prob=0.1,
        compute_dtype=jnp.bfloat16 if use_bf16 else None,
        learn_sigma=config.model.get("learn_sigma", False),
        rngs=nnx.Rngs(RNGManager(seed).next()),
    )


def _to_images(samples: jax.Array) -> list[np.ndarray]:
    """Convert normalized NHWC samples to Gradio-compatible grayscale images."""
    images = np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0)
    return [(image[..., 0] * 255.0).round().astype(np.uint8) for image in images]


def _make_generate(model: DiT, config: Any):
    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(x, t, y):
        output = model(x, t, y)
        if output.shape[-1] == x.shape[-1] * 2:
            return jnp.split(output, 2, axis=-1)[0]
        return output

    def generate(class_ids, num_samples: int, inference_steps: int, cfg_scale: float, seed: int):
        if not class_ids:
            raise ValueError("Select at least one class label")
        labels = jnp.asarray(
            [[int(class_id) for class_id in class_ids]] * num_samples,
            dtype=jnp.int32,
        )
        samples = sampler.sample_multi_conditional(
            model_fn,
            (
                num_samples,
                config.model.input_size,
                config.model.input_size,
                config.model.in_channels,
            ),
            jax.random.key(int(seed)),
            labels=labels,
            null_label=config.model.num_classes,
            num_inference_steps=int(inference_steps),
            cfg_scale=float(cfg_scale),
        )
        return _to_images(samples)

    return generate


def build_app(config_path: str, checkpoint: str, seed: int):
    """Build the Gradio application and restore the selected checkpoint once."""
    config = load_config(config_path)
    model = _build_demo_model(config, seed)
    _restore_ema(model, checkpoint)
    generate = _make_generate(model, config)

    def generate_with_caption(class_ids, num_samples, inference_steps, cfg_scale, sample_seed):
        images = generate(class_ids, num_samples, inference_steps, cfg_scale, sample_seed)
        names = ", ".join(CLASS_NAMES[int(class_id)] for class_id in class_ids)
        return images, f"Blended classes: {names}"

    with gr.Blocks(title="Fashion MNIST Diffusion") as app:
        gr.Markdown(
            "# Class-conditional Fashion MNIST\n"
            "Generate images from the TPU-v3 checkpoint. Select multiple classes "
            "to blend their classifier-free guidance directions."
        )
        gr.Markdown(
            "### Fashion MNIST class labels\n"
            + CLASS_LEGEND
        )
        with gr.Row():
            class_ids = gr.CheckboxGroup(
                choices=[(f"{index}: {name}", index) for index, name in enumerate(CLASS_NAMES)],
                value=[7],
                label="Class labels to blend",
            )
            num_samples = gr.Slider(1, 16, value=8, step=1, label="Number of samples")
        with gr.Row():
            inference_steps = gr.Slider(10, 100, value=50, step=5, label="Denoising steps")
            cfg_scale = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="Classifier-free guidance")
            sample_seed = gr.Slider(0, 100000, value=0, step=1, label="Random seed")
        generate_button = gr.Button("Generate samples", variant="primary")
        caption = gr.Markdown("Choose a class and generate samples.")
        gallery = gr.Gallery(label="Generated samples", columns=4, rows=4, height="auto")
        generate_button.click(
            generate_with_caption,
            inputs=[class_ids, num_samples, inference_steps, cfg_scale, sample_seed],
            outputs=[gallery, caption],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fashion_mnist_tpu.yaml")
    parser.add_argument(
        "--checkpoint",
        default="gs://diffjax/models/fashion-mnist_ccond_tpu-v3_12-08-2026/checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0, help="Model initialization seed.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Request a temporary public Gradio URL.")
    args = parser.parse_args()
    app = build_app(args.config, args.checkpoint, args.seed)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
