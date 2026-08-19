"""Unit tests for src/sampling/generator.py."""

from types import SimpleNamespace

from PIL import Image

from src.sampling.generator import generate_samples


def test_generate_samples_random_weights(tmp_path):
    """Verify standalone generation without checkpoint creates image grid."""
    config = SimpleNamespace(
        dataset="cifar10",
        model=SimpleNamespace(
            type="dit",
            input_size=8,
            patch_size=2,
            in_channels=3,
            hidden_size=32,
            depth=1,
            num_heads=2,
            num_classes=10,
            label_mode="class",
            label_dim=None,
            learn_sigma=False,
        ),
        training=SimpleNamespace(use_bf16=False),
        evaluation=SimpleNamespace(
            sample_count=4,
            num_inference_steps=2,
            cfg_scale=1.0,
        ),
    )

    out_file = tmp_path / "test_samples.png"
    images = generate_samples(
        config=config,
        checkpoint="",
        num_samples=4,
        num_inference_steps=2,
        output_path=str(out_file),
    )

    assert images.shape == (4, 8, 8, 3)
    assert out_file.is_file()
    saved_img = Image.open(out_file)
    assert saved_img.size == (16, 16)
