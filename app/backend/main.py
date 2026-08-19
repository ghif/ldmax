"""FastAPI web server for LDMAX model serving on Google Cloud Run."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Literal

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app.backend.config import settings  # noqa: E402
from app.backend.service import (  # noqa: E402
    CELEBA_ATTRIBUTE_NAMES,
    CIFAR10_CLASSES,
    FASHION_CLASSES,
    DiffusionService,
)

app = FastAPI(
    title="LDMAX Diffusion Service",
    description="FastAPI backend for serving class- and attribute-conditioned DiT models in JAX.",
    version="1.0.0",
)

# Configure Cross-Origin Resource Sharing (CORS) for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model service instance
service = DiffusionService(seed=settings.seed)


# ---------------------------------------------------------------------------
# Request and Response Schemas
# ---------------------------------------------------------------------------


class GenerateCIFAR10Request(BaseModel):
    """Request payload for CIFAR-10 generation."""

    class_weights: list[float] = Field(
        default=[1.0] + [0.0] * 9,
        description="Weights (0.0 - 1.0) for each of the 10 CIFAR-10 classes.",
    )
    num_samples: int = Field(default=8, ge=1, le=16)
    inference_steps: int = Field(default=50, ge=5, le=100)
    cfg_scale: float = Field(default=1.5, ge=1.0, le=10.0)
    seed: int = Field(default=0, ge=0)


class GenerateFashionMNISTRequest(BaseModel):
    """Request payload for Fashion-MNIST generation."""

    class_weights: list[float] = Field(
        default=[0.0] * 7 + [1.0] + [0.0] * 2,
        description="Weights (0.0 - 1.0) for each of the 10 Fashion-MNIST classes.",
    )
    num_samples: int = Field(default=8, ge=1, le=16)
    inference_steps: int = Field(default=50, ge=5, le=100)
    cfg_scale: float = Field(default=1.5, ge=1.0, le=10.0)
    seed: int = Field(default=0, ge=0)


class GenerateCelebARequest(BaseModel):
    """Request payload for CelebA latent diffusion generation."""

    selected_attributes: list[str] = Field(
        default=["Smiling", "Young"],
        description="List of binary facial attributes to condition on.",
    )
    num_samples: int = Field(default=4, ge=1, le=16)
    inference_steps: int = Field(default=50, ge=5, le=100)
    cfg_scale: float = Field(default=4.0, ge=1.0, le=10.0)
    seed: int = Field(default=42, ge=0)


class UnifiedGenerateRequest(BaseModel):
    """Unified request payload for all supported datasets."""

    dataset: Literal["cifar10", "fashion_mnist", "celeba"]
    class_weights: list[float] | None = None
    selected_attributes: list[str] | None = None
    num_samples: int = Field(default=8, ge=1, le=16)
    inference_steps: int = Field(default=50, ge=5, le=100)
    cfg_scale: float = Field(default=1.5, ge=1.0, le=10.0)
    seed: int = Field(default=0, ge=0)


class GenerateResponse(BaseModel):
    """Response payload containing base64 images and execution metadata."""

    images: list[str]
    caption: str
    time_taken_sec: float


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check():
    """Health check endpoint providing active JAX device platform information."""
    devices = [f"{d.platform}:{d.device_kind}" for d in jax.devices()]
    return {
        "status": "ok",
        "devices": devices,
        "default_platform": jax.devices()[0].platform,
    }


@app.get("/api/metadata")
def get_metadata():
    """Return dataset metadata, class names, and attributes for dynamic UI rendering."""
    return {
        "datasets": {
            "cifar10": {
                "name": "CIFAR-10",
                "resolution": "32×32",
                "mode": "RGB",
                "classes": CIFAR10_CLASSES,
                "default_active": 0,
                "default_cfg": 1.5,
            },
            "fashion_mnist": {
                "name": "Fashion-MNIST",
                "resolution": "28×28",
                "mode": "Grayscale",
                "classes": FASHION_CLASSES,
                "default_active": 7,
                "default_cfg": 1.5,
            },
            "celeba": {
                "name": "CelebA",
                "resolution": "256×256",
                "mode": "RGB (Latent Diffusion)",
                "attributes": list(CELEBA_ATTRIBUTE_NAMES),
                "default_active": ["Smiling", "Young"],
                "default_cfg": 4.0,
            },
        }
    }


@app.post("/api/generate/cifar10", response_model=GenerateResponse)
async def generate_cifar10(req: GenerateCIFAR10Request):
    """Generate CIFAR-10 images given class influences."""
    try:
        images, caption, elapsed = await asyncio.to_thread(
            service.generate_cifar10,
            config_path=settings.cifar10_config,
            checkpoint=settings.cifar10_checkpoint,
            class_weights=req.class_weights,
            num_samples=req.num_samples,
            inference_steps=req.inference_steps,
            cfg_scale=req.cfg_scale,
            seed=req.seed,
        )
        return GenerateResponse(images=images, caption=caption, time_taken_sec=elapsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/generate/fashion_mnist", response_model=GenerateResponse)
async def generate_fashion_mnist(req: GenerateFashionMNISTRequest):
    """Generate Fashion-MNIST images given class influences."""
    try:
        images, caption, elapsed = await asyncio.to_thread(
            service.generate_fashion_mnist,
            config_path=settings.fashion_config,
            checkpoint=settings.fashion_checkpoint,
            class_weights=req.class_weights,
            num_samples=req.num_samples,
            inference_steps=req.inference_steps,
            cfg_scale=req.cfg_scale,
            seed=req.seed,
        )
        return GenerateResponse(images=images, caption=caption, time_taken_sec=elapsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/generate/celeba", response_model=GenerateResponse)
async def generate_celeba(req: GenerateCelebARequest):
    """Generate CelebA faces given binary facial attributes."""
    try:
        images, caption, elapsed = await asyncio.to_thread(
            service.generate_celeba,
            config_path=settings.celeba_config,
            checkpoint=settings.celeba_checkpoint,
            selected_attributes=req.selected_attributes,
            num_samples=req.num_samples,
            inference_steps=req.inference_steps,
            cfg_scale=req.cfg_scale,
            seed=req.seed,
        )
        return GenerateResponse(images=images, caption=caption, time_taken_sec=elapsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/generate", response_model=GenerateResponse)
async def unified_generate(req: UnifiedGenerateRequest):
    """Unified generation endpoint supporting all three datasets."""
    try:
        if req.dataset == "cifar10":
            class_weights = req.class_weights or ([1.0] + [0.0] * 9)
            images, caption, elapsed = await asyncio.to_thread(
                service.generate_cifar10,
                config_path=settings.cifar10_config,
                checkpoint=settings.cifar10_checkpoint,
                class_weights=class_weights,
                num_samples=req.num_samples,
                inference_steps=req.inference_steps,
                cfg_scale=req.cfg_scale,
                seed=req.seed,
            )
        elif req.dataset == "fashion_mnist":
            class_weights = req.class_weights or ([0.0] * 7 + [1.0] + [0.0] * 2)
            images, caption, elapsed = await asyncio.to_thread(
                service.generate_fashion_mnist,
                config_path=settings.fashion_config,
                checkpoint=settings.fashion_checkpoint,
                class_weights=class_weights,
                num_samples=req.num_samples,
                inference_steps=req.inference_steps,
                cfg_scale=req.cfg_scale,
                seed=req.seed,
            )
        elif req.dataset == "celeba":
            selected_attrs = req.selected_attributes or ["Smiling", "Young"]
            images, caption, elapsed = await asyncio.to_thread(
                service.generate_celeba,
                config_path=settings.celeba_config,
                checkpoint=settings.celeba_checkpoint,
                selected_attributes=selected_attrs,
                num_samples=req.num_samples,
                inference_steps=req.inference_steps,
                cfg_scale=req.cfg_scale,
                seed=req.seed,
            )
        else:
            raise ValueError(f"Unknown dataset: {req.dataset}")

        return GenerateResponse(images=images, caption=caption, time_taken_sec=elapsed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
