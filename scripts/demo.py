"""Unified interactive Gradio demo for CIFAR-10 and Fashion MNIST sampling."""

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised by the CLI
    gr = None

from flax import nnx  # noqa: E402

from src.models.dit.dit import DiT, resolve_conditioning_mode  # noqa: E402
from src.sampling.cifar10 import _restore_ema as restore_cifar10_ema  # noqa: E402
from src.sampling.fashion_mnist import _restore_ema as restore_fashion_ema  # noqa: E402
from src.training.sampler import DDIMSampler  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.rng import RNGManager  # noqa: E402

CIFAR10_CLASSES = [
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

FASHION_CLASSES = [
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


def _build_demo_model(config: Any, seed: int) -> DiT:
    """Build the checkpoint-compatible model for the active JAX backend."""
    conditioning = config.model.get("conditioning", "class")
    if conditioning != "class":
        raise ValueError("This demo requires a class-conditioned model")
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


def _to_rgb_images(samples: jax.Array) -> list[np.ndarray]:
    """Convert normalized NHWC samples to Gradio-compatible RGB images."""
    images = np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0)
    return [(image[..., :3] * 255.0).round().astype(np.uint8) for image in images]


def _to_grayscale_images(samples: jax.Array) -> list[np.ndarray]:
    """Convert normalized NHWC samples to Gradio-compatible grayscale images."""
    images = np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0)
    return [(image[..., 0] * 255.0).round().astype(np.uint8) for image in images]


def _make_generate(model: DiT, config: Any, is_grayscale: bool = False):
    sampler = DDIMSampler()

    @nnx.jit
    def model_fn(x, t, y):
        output = model(x, t, y)
        if output.shape[-1] == x.shape[-1] * 2:
            return jnp.split(output, 2, axis=-1)[0]
        return output

    def generate(
        class_weights,
        num_samples: int,
        inference_steps: int,
        cfg_scale: float,
        seed: int,
    ):
        positive_classes = [
            index for index, weight in enumerate(class_weights) if float(weight) > 0
        ]
        if not positive_classes:
            raise ValueError("Give at least one class a positive influence")
        labels = jnp.asarray(
            [positive_classes] * num_samples,
            dtype=jnp.int32,
        )
        weights = jnp.asarray(
            [float(class_weights[index]) for index in positive_classes],
            dtype=jnp.float32,
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
            weights=weights,
            num_inference_steps=int(inference_steps),
            cfg_scale=float(cfg_scale),
            clip_denoised=not is_grayscale,
        )
        return _to_grayscale_images(samples) if is_grayscale else _to_rgb_images(samples)

    return generate


def _build_dataset_tab(
    dataset_name: str,
    class_names: list[str],
    generate_fn: Any,
    default_active_idx: int = 0,
    model_description: str = "",
) -> None:
    """Build the UI layout and event listeners for a single dataset tab."""
    gr.Markdown(f"### Model configuration\n{model_description}")
    legend = " | ".join(f"**{index}** {name}" for index, name in enumerate(class_names))
    gr.Markdown(f"### {dataset_name} class labels\n{legend}")

    def generate_with_caption(*values):
        class_weights = values[: len(class_names)]
        num_samples, inference_steps, cfg_scale, sample_seed = values[len(class_names) :]
        images = generate_fn(class_weights, num_samples, inference_steps, cfg_scale, sample_seed)
        active = [
            f"{class_names[index]} ({float(weight):.2f})"
            for index, weight in enumerate(class_weights)
            if float(weight) > 0
        ]
        return images, "Influences: " + ", ".join(active)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### Controls\nSet the relative influence for each class.")
            class_sliders = []
            for index, name in enumerate(class_names):
                class_sliders.append(
                    gr.Slider(
                        0.0,
                        1.0,
                        value=1.0 if index == default_active_idx else 0.0,
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


def build_app(
    cifar10_config_path: str,
    cifar10_checkpoint: str,
    fashion_config_path: str,
    fashion_checkpoint: str,
    seed: int = 0,
):
    """Build the unified multi-tab Gradio application."""
    if gr is None:
        raise RuntimeError(
            "Gradio is required for this demo. Install the project dependencies "
            "or run: python -m pip install gradio"
        )

    # CIFAR-10 model setup
    cifar_config = load_config(cifar10_config_path)
    cifar_model = _build_demo_model(cifar_config, seed)
    restore_cifar10_ema(cifar_model, cifar10_checkpoint)
    cifar_generate = _make_generate(cifar_model, cifar_config, is_grayscale=False)

    # Fashion MNIST model setup
    fashion_config = load_config(fashion_config_path)
    fashion_model = _build_demo_model(fashion_config, seed)
    restore_fashion_ema(fashion_model, fashion_checkpoint)
    fashion_generate = _make_generate(fashion_model, fashion_config, is_grayscale=True)

    with gr.Blocks(title="LDMAX Diffusion Image Generator") as app:
        gr.Markdown(
            "# LDMAX Diffusion Image Generator\n"
            "Generate images with class-conditioned Diffusion Transformer (**DiT**) "
            "models trained on JAX / TPU. Switch between the tabs below to explore datasets."
        )

        with gr.Tabs():
            with gr.Tab("CIFAR-10 (32×32 RGB)"):
                _build_dataset_tab(
                    dataset_name="CIFAR-10",
                    class_names=CIFAR10_CLASSES,
                    generate_fn=cifar_generate,
                    default_active_idx=0,
                    model_description=(
                        "Raw-pixel **DiT** (8 blocks, 256 hidden size, 8 heads, 2×2 patches). "
                        "Trained on **CIFAR-10 (32×32×3)** on TPU using BF16 compute."
                    ),
                )
            with gr.Tab("Fashion-MNIST (28×28 Grayscale)"):
                _build_dataset_tab(
                    dataset_name="Fashion-MNIST",
                    class_names=FASHION_CLASSES,
                    generate_fn=fashion_generate,
                    default_active_idx=7,
                    model_description=(
                        "Raw-pixel **DiT** (6 blocks, 192 hidden size, 6 heads, 2×2 patches). "
                        "Trained on **Fashion-MNIST (28×28×1)** for 30,000 steps on TPU v6e-1."
                    ),
                )

    return app


def main() -> None:
    """Launch the unified multi-dataset Gradio web demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar10-config", default="configs/cifar10_pixel.yaml")
    parser.add_argument(
        "--cifar10-checkpoint",
        default="gs://diffjax/models/cifar10_pixel_ccond_tpu_15-08-2026/checkpoints",
    )
    parser.add_argument("--fashion-config", default="configs/fashion_mnist_tpu_v4.yaml")
    parser.add_argument(
        "--fashion-checkpoint",
        default="gs://diffjax/models/fashion-mnist_ccond_tpu-v4_12-08-2026/checkpoints",
    )
    parser.add_argument("--seed", type=int, default=0, help="Model initialization seed.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Request a temporary public Gradio URL."
    )
    args = parser.parse_args()

    app = build_app(
        cifar10_config_path=args.cifar10_config,
        cifar10_checkpoint=args.cifar10_checkpoint,
        fashion_config_path=args.fashion_config,
        fashion_checkpoint=args.fashion_checkpoint,
        seed=args.seed,
    )
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
