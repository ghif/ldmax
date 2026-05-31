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

### 1. Training

LDMAX supports training in the latent space for multiple datasets:

**CIFAR-10 ($32 \times 32 \to 4 \times 4$ latents):**
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m src.scripts.train --config configs/cifar10.yaml --output_dir ./outputs/cifar
```

**CelebA ($256 \times 256 \to 32 \times 32$ latents):**
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m src.scripts.train --config configs/celeba.yaml --output_dir ./outputs/celeba
```

### 2. Monitoring

Track loss and view periodic visual samples in TensorBoard:
```bash
tensorboard --logdir ./outputs
```

### 3. Standalone Sampling

Generate high-quality samples from a saved checkpoint:
```bash
export PYTHONPATH=$PYTHONPATH:.
python -m src.scripts.sample \
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
└── scripts/            # CLI entry points for train/sample
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
