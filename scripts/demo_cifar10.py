"""Interactive Gradio demo for class-conditional CIFAR-10 sampling."""

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised by the CLI
    gr = None

from flax import nnx

from src.models.dit.dit import DiT, resolve_conditioning_mode
from src.sampling.cifar10 import _restore_ema
from src.training.sampler import DDIMSampler
from src.utils.config import load_config
from src.utils.rng import RNGManager


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CLASS_LEGEND = " | ".join(
    f"**{index}** {name}" for index, name in enumerate(CLASS_NAMES)
)


def _build_demo_model(config: Any, seed: int) -> DiT:
    """Build the checkpoint-compatible model for the active JAX backend."""
    conditioning = config.model.get("conditioning", "class")
    if conditioning != "class":
        raise ValueError("This demo requires a class-conditioned CIFAR-10 model")
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
    """Convert normalized NHWC samples to Gradio-compatible RGB images."""
    images = np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0)
    return [(image[..., :3] * 255.0).round().astype(np.uint8) for image in images]


def _make_generate(model: DiT, config: Any):
    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(x, t, y):
        output = model(x, t, y)
        if output.shape[-1] == x.shape[-1] * 2:
            return jnp.split(output, 2, axis=-1)[0]
        return output

    def generate(class_weights, num_samples: int, inference_steps: int, cfg_scale: float, seed: int):
        positive_classes = [index for index, weight in enumerate(class_weights) if float(weight) > 0]
        if not positive_classes:
            raise ValueError("Give at least one class a positive influence")
        labels = jnp.asarray(
            [positive_classes] * num_samples,
            dtype=jnp.int32,
        )
        weights = jnp.asarray([float(class_weights[index]) for index in positive_classes], dtype=jnp.float32)
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
            weights=weights,
            num_inference_steps=int(inference_steps),
            cfg_scale=float(cfg_scale),
            clip_denoised=True,
        )
        return _to_images(samples)

    return generate


def build_app(config_path: str, checkpoint: str, seed: int):
    """Build the Gradio application and restore the selected checkpoint once."""
    if gr is None:
        raise RuntimeError(
            "Gradio is required for this demo. Install the project dependencies "
            "or run: python -m pip install gradio"
        )
    config = load_config(config_path)
    model = _build_demo_model(config, seed)
    _restore_ema(model, checkpoint)
    generate = _make_generate(model, config)

    def generate_with_caption(*values):
        class_weights = values[: len(CLASS_NAMES)]
        num_samples, inference_steps, cfg_scale, sample_seed = values[len(CLASS_NAMES) :]
        images = generate(class_weights, num_samples, inference_steps, cfg_scale, sample_seed)
        active = [
            f"{CLASS_NAMES[index]} ({float(weight):.2f})"
            for index, weight in enumerate(class_weights)
            if float(weight) > 0
        ]
        return images, "Influences: " + ", ".join(active)

    with gr.Blocks(title="CIFAR-10 Diffusion Image Generator") as app:
        gr.Markdown(
            "# CIFAR-10 Diffusion Image Generator\n"
            "Generate **32×32 RGB CIFAR-10 images** with a class-conditioned "
            "diffusion model. The model starts from random noise and progressively "
            "denoises it into an image guided by the selected class labels. Adjust "
            "each class influence to generate a single category or blend multiple "
            "categories."
        )
        gr.Markdown(
            "### Model configuration\n"
            "This demo uses a raw-pixel **DiT (Diffusion Transformer)** model, so no "
            "VAE encoder or decoder is involved. The network has **8 transformer "
            "blocks**, a **256-dimensional hidden size**, **8 attention heads**, "
            "2×2 input patches, and 10 class labels. It was trained on **CIFAR-10 "
            "(32×32×3)** on a **Google Cloud TPU**, using BF16 activation compute "
            "with FP32-sensitive parameters and statistics preserved."
        )
        gr.Markdown(
            "### CIFAR-10 class labels\n"
            + CLASS_LEGEND
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Controls\nSet the relative influence for each class.")
                class_sliders = []
                for index, name in enumerate(CLASS_NAMES):
                    class_sliders.append(
                        gr.Slider(
                            0.0,
                            1.0,
                            value=1.0 if index == 0 else 0.0,
                            step=0.05,
                            label=f"{index}: {name}",
                            scale=1,
                            min_width=180,
                        )
                    )
                num_samples = gr.Slider(
                    1, 16, value=8, step=1, label="Number of samples", scale=1, min_width=180
                )
                inference_steps = gr.Slider(
                    10, 100, value=50, step=5, label="Denoising steps", scale=1, min_width=180
                )
                cfg_scale = gr.Slider(
                    1.0,
                    5.0,
                    value=1.5,
                    step=0.1,
                    label="Classifier-free guidance",
                    scale=1,
                    min_width=180,
                )
                sample_seed = gr.Slider(
                    0, 100000, value=0, step=1, label="Random seed", scale=1, min_width=180
                )
                generate_button = gr.Button("Generate samples", variant="primary")

            with gr.Column(scale=2, min_width=520):
                caption = gr.Markdown("Choose class influences and generate samples.")
                gallery = gr.Gallery(
                    label="Generated samples",
                    columns=4,
                    rows=4,
                    height=720,
                )
        generate_button.click(
            generate_with_caption,
            inputs=class_sliders + [num_samples, inference_steps, cfg_scale, sample_seed],
            outputs=[gallery, caption],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar10_pixel.yaml")
    parser.add_argument(
        "--checkpoint",
        default="gs://diffjax/models/cifar10_pixel_ccond_tpu_15-08-2026/checkpoints",
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
