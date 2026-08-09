# LDMAX: Latent Diffusion Models in JAX

![JAX](https://img.shields.io/badge/JAX-Enabled-blue.svg)
![Flax NNX](https://img.shields.io/badge/Flax-NNX-orange.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**LDMAX** is an educational and collaborative AI research repository focused on training **Latent Diffusion Models (LDM)** from scratch. Built entirely on the modern JAX ecosystem, it serves as a highly readable, modular, and performant foundation for generative model research.

## 🌟 Core Principles

LDMAX adheres to five foundational tenets:
1. **Educational First & Clear Documentation**: Code is extensively documented for clarity and pedagogical value.
2. **Clean Code Architecture**: Strictly modular design using SOLID principles and dependency injection.
3. **Collaborative AI Research**: Architecture-agnostic structure allows easy swapping of models and datasets.
4. **Visual Generative Focus**: Tooling optimized for spatial data and visual metrics (FID).
5. **Deterministic Reproducibility**: Enforces fixed random seeds and explicit environments.

## 🚀 Key Features

- **Live VAE Encoding**: JIT-compiled encoding of images into latent space during training using pre-trained VAEs (e.g., Stable Diffusion's autoencoder).
- **Flax NNX**: Leverages the new reference-based Flax API for intuitive model management.
- **Grain Data Pipelines**: High-performance deterministic data loading using `google-grain` and Hugging Face `datasets`.
- **Architecture Agnostic**: Core logic is separated from specific model implementations, allowing easy integration of DiT, UNets, or new variants.
- **Advanced Training**: Includes `bfloat16` mixed precision, Exponential Moving Average (EMA) weight tracking, and Orbax async checkpointing.

## 🛠️ Installation

```bash
# 1. Setup Conda Environment
conda create -n ldmax python=3.11 -y
conda activate ldmax

# 2. Install JAX (Modify based on hardware: CPU/GPU/TPU)
conda install -y -c conda-forge jax jaxlib flax optax

# 3. Install remaining dependencies
pip install -r requirements.txt
```

## 🏃‍♂️ Quickstart

### Training

LDMAX supports training in the latent space for multiple datasets:

**Fashion MNIST raw-pixel diffusion:**

This path trains DiT directly on `28 × 28 × 1` grayscale images. It does not
load or call the latent VAE encoder or decoder.

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m scripts.train_fashion_mnist \
    --config configs/fashion_mnist.yaml \
    --output_dir ./outputs/fashion_mnist
```

For a two-step smoke run, use `configs/fashion_mnist_test.yaml`.

To sample from a saved Fashion MNIST checkpoint and write a grayscale image
grid:

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m scripts.sample_fashion_mnist \
    --config configs/fashion_mnist.yaml \
    --checkpoint ./outputs/fashion_mnist/checkpoints/1000 \
    --num_samples 16 \
    --output_path ./samples/fashion_mnist.png
```

**CIFAR-10 ($128 \times 128 \to 16 \times 16$ latents):**
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m scripts.train --config configs/cifar10.yaml --output_dir ./outputs/cifar
```

**CelebA ($256 \times 256 \to 32 \times 32$ latents):**
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m scripts.train --config configs/celeba.yaml --output_dir ./outputs/celeba
```

**CelebA TPU v6e batch sweep:**
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m scripts.train --config configs/celeba_tpu_b128.yaml --output_dir ./outputs/celeba_b128
python -m scripts.train --config configs/celeba_tpu_b256.yaml --output_dir ./outputs/celeba_b256
python -m scripts.train --config configs/celeba_tpu_b384.yaml --output_dir ./outputs/celeba_b384
```
Use the run with the best samples/sec and stable loss as the final CelebA TPU configuration.

### 2. Monitoring

Track loss and view periodic visual samples in TensorBoard:
```bash
tensorboard --logdir ./outputs
```

### 3. Standalone Sampling

Generate high-quality samples from a saved checkpoint:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m scripts.sample \
    --checkpoint ./outputs/celeba/checkpoints/50000 \
    --num_samples 16 \
    --num_steps 50 \
    --output_path ./samples.png
```

## 📁 Project Structure

```text
src/
├── models/             # Architecture implementations (e.g., dit/)
├── data/               # Dataset loaders (CIFAR, CelebA) via Grain
├── training/           # Shared training steps, samplers, and EMA
├── utils/              # VAE, Checkpointing (Orbax), Logging, RNG
└── scripts/            # Thin CLI launchers only
tests/                  # Unit & Integration tests
configs/                # YAML experiment definitions
```

## 🧪 Verification

LDMAX includes a comprehensive test suite covering data loading, model forward passes, and VAE reconstruction.

```bash
# Run all tests
export PYTHONPATH=$PYTHONPATH:.
pytest tests/

# Visual verification (checks VAE reconstruction quality)
pytest -s tests/unit/test_vae.py
```

## 📚 References

- **Scalable Diffusion Models with Transformers (DiT)** ([Peebles et al. 2023](https://arxiv.org/abs/2212.09748))
- **Flax NNX** ([Documentation](https://flax.readthedocs.io/en/latest/nnx/index.html))
- **Hugging Face Datasets** ([cifar10](https://huggingface.co/datasets/uoft-cs/cifar10), [celeba](https://huggingface.co/datasets/nielsr/CelebA-faces))
