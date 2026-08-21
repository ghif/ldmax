"""Model management and inference service for the LDMAX backend."""

from __future__ import annotations

import base64
import gc
import io
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from orbax.checkpoint import type_handlers
from PIL import Image

from src.data.celeba import CELEBA_ATTRIBUTE_NAMES
from src.models.dit.dit import DiT
from src.models.factory import create_model
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager, materialize_checkpoint
from src.utils.config import load_config
from src.utils.rng import RNGManager
from src.utils.vae import VAEManager

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


def _restore_model_ema(model: DiT, checkpoint: str) -> None:
    """Restore EMA parameters from an individual Orbax checkpoint directory."""
    checkpoint_path = materialize_checkpoint(checkpoint)
    checkpoint_root = checkpoint_path.parent
    checkpoint_step = int(checkpoint_path.name)
    manager = CheckpointManager(checkpoint_root)
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    pure_state = nnx.state(model).to_pure_dict()

    def template(value):
        if isinstance(value, dict):
            return {key: template(child) for key, child in value.items()}
        return {"value": jax.device_put(value, sharding)}

    restore_template = {"ema": template(pure_state)}
    restore_args = jax.tree.map(
        lambda _: type_handlers.ArrayRestoreArgs(sharding=sharding),
        restore_template,
    )
    state = manager.restore(
        checkpoint_step,
        args=ocp.args.PyTreeRestore(
            restore_template,
            restore_args=restore_args,
            partial_restore=True,
        ),
    )
    if state is None or "ema" not in state:
        raise ValueError(f"Checkpoint does not contain EMA parameters: {checkpoint}")

    checkpoint_state = state["ema"]
    flat_state = nnx.state(model).flat_state()
    for path, variable in zip(flat_state.paths, flat_state.leaves):
        value = checkpoint_state
        for key in path:
            if not isinstance(value, dict):
                raise ValueError(f"Malformed EMA state at path {path}")
            value = value[key] if key in value else value[str(key)]
        if isinstance(value, dict) and "value" in value:
            value = value["value"]
        variable.value = value

    del state, checkpoint_state, restore_template, restore_args, pure_state
    gc.collect()


def _image_to_base64_png(image_array: np.ndarray) -> str:
    """Convert a uint8 numpy image (H, W) or (H, W, 3) to a base64 encoded data URI."""
    if image_array.ndim == 2:
        img = Image.fromarray(image_array, mode="L")
    elif image_array.ndim == 3 and image_array.shape[2] == 1:
        img = Image.fromarray(image_array[:, :, 0], mode="L")
    else:
        img = Image.fromarray(image_array, mode="RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


class DiffusionService:
    """Singleton service that caches loaded models and executes fast DDIM sampling."""

    def __init__(self, seed: int = 0):
        """Initialize the diffusion service."""
        self.seed = seed
        self.sampler = DDIMSampler()
        self._models: dict[str, DiT] = {}
        self._configs: dict[str, Any] = {}
        self._vae_manager: VAEManager | None = None

    def get_or_load_model(
        self, dataset: str, config_path: str, checkpoint: str
    ) -> tuple[DiT, Any]:
        """Lazy-load and cache the requested model and its config."""
        if dataset not in self._models:
            print(f"Loading {dataset} model from {checkpoint}...")
            config = load_config(config_path)
            use_bf16 = (
                config.training.get("use_bf16", False)
                and jax.devices()[0].platform == "tpu"
            )
            config.training.use_bf16 = use_bf16

            model = create_model(config, RNGManager(self.seed).next())
            if checkpoint:
                _restore_model_ema(model, checkpoint)

            self._models[dataset] = model
            self._configs[dataset] = config

        return self._models[dataset], self._configs[dataset]

    def get_vae_manager(self) -> VAEManager:
        """Lazy-load the VAE manager for latent decoding."""
        if self._vae_manager is None:
            print("Loading pretrained VAE manager...")
            self._vae_manager = VAEManager()
        return self._vae_manager

    def generate_cifar10(
        self,
        config_path: str,
        checkpoint: str,
        class_weights: list[float],
        num_samples: int = 8,
        inference_steps: int = 50,
        cfg_scale: float = 1.5,
        seed: int = 0,
    ) -> tuple[list[str], str, float]:
        """Generate class-conditioned CIFAR-10 images."""
        start_time = time.perf_counter()
        model, config = self.get_or_load_model("cifar10", config_path, checkpoint)

        positive_classes = [
            idx for idx, weight in enumerate(class_weights) if float(weight) > 0
        ]
        if not positive_classes:
            raise ValueError("Give at least one class a positive influence")

        labels = jnp.asarray([positive_classes] * num_samples, dtype=jnp.int32)
        weights = jnp.asarray(
            [float(class_weights[idx]) for idx in positive_classes], dtype=jnp.float32
        )

        @nnx.jit
        def model_fn(x, t, y):
            out = model(x, t, y)
            if out.shape[-1] == x.shape[-1] * 2:
                return jnp.split(out, 2, axis=-1)[0]
            return out

        shape = (
            num_samples,
            config.model.input_size,
            config.model.input_size,
            config.model.in_channels,
        )
        samples = self.sampler.sample_multi_conditional(
            model_fn,
            shape,
            jax.random.key(int(seed)),
            labels=labels,
            null_label=config.model.num_classes,
            weights=weights,
            num_inference_steps=int(inference_steps),
            cfg_scale=float(cfg_scale),
            clip_denoised=True,
        )

        # Convert to RGB uint8
        images_np = np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0)
        rgb_list = [
            (img[..., :3] * 255.0).round().astype(np.uint8) for img in images_np
        ]
        base64_images = [_image_to_base64_png(img) for img in rgb_list]

        active_str = ", ".join(
            f"{CIFAR10_CLASSES[i]} ({float(class_weights[i]):.2f})"
            for i in positive_classes
        )
        caption = f"Influences: {active_str}"
        elapsed = time.perf_counter() - start_time
        del samples, images_np, rgb_list
        gc.collect()
        return base64_images, caption, elapsed

    def generate_fashion_mnist(
        self,
        config_path: str,
        checkpoint: str,
        class_weights: list[float],
        num_samples: int = 8,
        inference_steps: int = 50,
        cfg_scale: float = 1.5,
        seed: int = 0,
    ) -> tuple[list[str], str, float]:
        """Generate class-conditioned Fashion-MNIST images."""
        start_time = time.perf_counter()
        model, config = self.get_or_load_model(
            "fashion_mnist", config_path, checkpoint
        )

        positive_classes = [
            idx for idx, weight in enumerate(class_weights) if float(weight) > 0
        ]
        if not positive_classes:
            raise ValueError("Give at least one class a positive influence")

        labels = jnp.asarray([positive_classes] * num_samples, dtype=jnp.int32)
        weights = jnp.asarray(
            [float(class_weights[idx]) for idx in positive_classes], dtype=jnp.float32
        )

        @nnx.jit
        def model_fn(x, t, y):
            out = model(x, t, y)
            if out.shape[-1] == x.shape[-1] * 2:
                return jnp.split(out, 2, axis=-1)[0]
            return out

        shape = (
            num_samples,
            config.model.input_size,
            config.model.input_size,
            config.model.in_channels,
        )
        samples = self.sampler.sample_multi_conditional(
            model_fn,
            shape,
            jax.random.key(int(seed)),
            labels=labels,
            null_label=config.model.num_classes,
            weights=weights,
            num_inference_steps=int(inference_steps),
            cfg_scale=float(cfg_scale),
            clip_denoised=False,
        )

        images_np = np.asarray((samples + 1.0).clip(0.0, 2.0) / 2.0)
        gray_list = [
            (img[..., 0] * 255.0).round().astype(np.uint8) for img in images_np
        ]
        base64_images = [_image_to_base64_png(img) for img in gray_list]

        active_str = ", ".join(
            f"{FASHION_CLASSES[i]} ({float(class_weights[i]):.2f})"
            for i in positive_classes
        )
        caption = f"Influences: {active_str}"
        elapsed = time.perf_counter() - start_time
        del samples, images_np, gray_list
        gc.collect()
        return base64_images, caption, elapsed

    def generate_celeba(
        self,
        config_path: str,
        checkpoint: str,
        selected_attributes: list[str],
        num_samples: int = 4,
        inference_steps: int = 50,
        cfg_scale: float = 4.0,
        seed: int = 42,
    ) -> tuple[list[str], str, float]:
        """Generate attribute-conditioned CelebA 256x256 RGB images via Latent Diffusion."""
        start_time = time.perf_counter()
        model, config = self.get_or_load_model("celeba", config_path, checkpoint)
        vae = self.get_vae_manager()

        num_attrs = getattr(config.model, "label_dim", 40)
        y = np.zeros((num_samples, num_attrs), dtype=np.float32)
        if selected_attributes:
            for attr in selected_attributes:
                if attr in CELEBA_ATTRIBUTE_NAMES:
                    idx = CELEBA_ATTRIBUTE_NAMES.index(attr)
                    y[:, idx] = 1.0

        y_tensor = jnp.asarray(y)
        null_y = jnp.zeros_like(y_tensor)

        @nnx.jit
        def model_fn(x, t, y):
            out = model(x, t, y)
            if out.shape[-1] == x.shape[-1] * 2:
                return jnp.split(out, 2, axis=-1)[0]
            return out

        latents = self.sampler.sample(
            model_fn=model_fn,
            shape=(
                num_samples,
                config.model.input_size,
                config.model.input_size,
                config.model.in_channels,
            ),
            rng_key=jax.random.key(int(seed)),
            num_inference_steps=int(inference_steps),
            y=y_tensor,
            null_y=null_y,
            cfg_scale=float(cfg_scale),
        )

        decoded_images = vae.decode(latents)
        rgb_list = [
            (img * 255.0).round().astype(np.uint8) for img in np.asarray(decoded_images)
        ]
        base64_images = [_image_to_base64_png(img) for img in rgb_list]

        active_str = (
            ", ".join(selected_attributes)
            if selected_attributes
            else "None (unconditioned)"
        )
        caption = f"Active attributes: {active_str}"
        elapsed = time.perf_counter() - start_time
        del latents, decoded_images, rgb_list, y, y_tensor, null_y
        gc.collect()
        return base64_images, caption, elapsed
