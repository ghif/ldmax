"""Unit tests for the FastAPI backend app."""

import pytest
from fastapi.testclient import TestClient

FAKE_B64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def client():
    """Create FastAPI test client."""
    from app.backend.main import app

    return TestClient(app)


def test_health_check(client):
    """Test /api/health endpoint returns platform and status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "default_platform" in data
    assert "devices" in data


def test_metadata_endpoint(client):
    """Test /api/metadata endpoint returns dataset information."""
    response = client.get("/api/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "datasets" in data
    assert "cifar10" in data["datasets"]
    assert "fashion_mnist" in data["datasets"]
    assert "celeba" in data["datasets"]
    assert len(data["datasets"]["cifar10"]["classes"]) == 10
    assert len(data["datasets"]["fashion_mnist"]["classes"]) == 10
    assert len(data["datasets"]["celeba"]["attributes"]) == 40


def test_generate_cifar10(client, monkeypatch):
    """Test /api/generate/cifar10 with mocked service."""
    from app.backend.main import service

    monkeypatch.setattr(
        service,
        "generate_cifar10",
        lambda *args, **kwargs: ([FAKE_B64] * 2, "Influences: airplane (1.00)", 0.25),
    )

    response = client.post(
        "/api/generate/cifar10",
        json={
            "class_weights": [1.0] + [0.0] * 9,
            "num_samples": 2,
            "inference_steps": 10,
            "cfg_scale": 1.5,
            "seed": 0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) == 2
    assert "airplane" in data["caption"]
    assert data["time_taken_sec"] == 0.25


def test_generate_fashion_mnist(client, monkeypatch):
    """Test /api/generate/fashion_mnist with mocked service."""
    from app.backend.main import service

    monkeypatch.setattr(
        service,
        "generate_fashion_mnist",
        lambda *args, **kwargs: ([FAKE_B64] * 2, "Influences: Sneaker (1.00)", 0.20),
    )

    response = client.post(
        "/api/generate/fashion_mnist",
        json={
            "class_weights": [0.0] * 7 + [1.0] + [0.0] * 2,
            "num_samples": 2,
            "inference_steps": 10,
            "cfg_scale": 1.5,
            "seed": 0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) == 2
    assert "Sneaker" in data["caption"]


def test_generate_celeba(client, monkeypatch):
    """Test /api/generate/celeba with mocked service."""
    from app.backend.main import service

    monkeypatch.setattr(
        service,
        "generate_celeba",
        lambda *args, **kwargs: ([FAKE_B64] * 2, "Active attributes: Smiling, Young", 0.45),
    )

    response = client.post(
        "/api/generate/celeba",
        json={
            "selected_attributes": ["Smiling", "Young"],
            "num_samples": 2,
            "inference_steps": 10,
            "cfg_scale": 4.0,
            "seed": 42,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) == 2
    assert "Smiling" in data["caption"]


def test_unified_generate(client, monkeypatch):
    """Test /api/generate unified endpoint."""
    from app.backend.main import service

    monkeypatch.setattr(
        service,
        "generate_cifar10",
        lambda *args, **kwargs: ([FAKE_B64], "caption", 0.1),
    )

    response = client.post(
        "/api/generate",
        json={
            "dataset": "cifar10",
            "class_weights": [1.0] + [0.0] * 9,
            "num_samples": 1,
            "inference_steps": 10,
        },
    )
    assert response.status_code == 200
    assert response.json()["images"] == [FAKE_B64]
