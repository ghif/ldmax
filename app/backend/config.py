"""Configuration settings for the LDMAX FastAPI backend."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Backend server and model configuration."""

    # Server settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    cors_origins: list[str] = None

    # Model configuration paths
    cifar10_config: str = os.getenv("CIFAR10_CONFIG", "configs/cifar10_pixel.yaml")
    cifar10_checkpoint: str = os.getenv(
        "CIFAR10_CHECKPOINT",
        "gs://diffjax/models/cifar10_pixel_ccond_tpu_15-08-2026/checkpoints",
    )

    fashion_config: str = os.getenv(
        "FASHION_CONFIG", "configs/fashion_mnist_tpu_v4.yaml"
    )
    fashion_checkpoint: str = os.getenv(
        "FASHION_CHECKPOINT",
        "gs://diffjax/models/fashion-mnist_ccond_tpu-v4_12-08-2026/checkpoints",
    )

    celeba_config: str = os.getenv("CELEBA_CONFIG", "configs/celeba.yaml")
    celeba_checkpoint: str = os.getenv(
        "CELEBA_CHECKPOINT",
        "gs://diffjax/models/celeba_ldm_ccond_tpu-v6e-1_18-08-2026/checkpoints/270000",
    )

    # Execution settings
    use_bf16_on_tpu: bool = True
    seed: int = int(os.getenv("SEED", "0"))

    def __post_init__(self):
        """Initialize and parse CORS origins list."""
        if self.cors_origins is None:
            raw_cors = os.getenv("CORS_ORIGINS", "*")
            self.cors_origins = [
                origin.strip() for origin in raw_cors.split(",") if origin.strip()
            ]


settings = Settings()
