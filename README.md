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

# 2. Install dependencies for your target accelerator.
#    Choose one: requirements_cpu.txt, requirements_gpu.txt, or requirements_tpu.txt.
pip install -r requirements_cpu.txt

# 3. For GPU or TPU, replace the previous command with the matching file:
# pip install -r requirements_gpu.txt
# pip install -r requirements_tpu.txt
```

The Fashion MNIST runner is single-device and does not require a mesh, Grain,
or accelerator-specific model code. With one visible local device, JAX places
the model and computations on that device automatically. Select the backend
through the JAX installation and runtime environment, for example with
`JAX_PLATFORMS=cpu` when explicitly testing CPU execution.

The TPU Fashion MNIST configuration uses conservative BF16 mixed precision:
model parameters, optimizer state, EMA weights, normalization, diffusion
schedules, and loss reductions remain FP32, while activations and matrix
multiplications use BF16. This preserves the quality of the FP32 path while
allowing TPU acceleration.

## 🏃‍♂️ Quickstart

### Training

LDMAX supports training in the latent space for multiple datasets:

**Fashion MNIST raw-pixel diffusion:**

This path trains DiT directly on `28 × 28 × 1` grayscale images. It does not
load or call the latent VAE encoder or decoder. To keep the introductory data
path easy to follow, it uses Hugging Face Datasets with simple NumPy batching;
the larger CIFAR-10 and CelebA pipelines use Grain for higher-throughput data
loading and sharding.

```bash
export PYTHONPATH=$PYTHONPATH:.
python -m scripts.train_fashion_mnist \
    --config configs/fashion_mnist.yaml \
    --output_dir ./outputs/fashion_mnist
```

For a two-step smoke run, use `configs/fashion_mnist_test.yaml`.

**CIFAR10 native-pixel diffusion:**

The native-pixel CIFAR10 workflow trains directly on normalized `32 × 32 × 3`
RGB images and does not load or call a VAE. It supports both class-conditional
and unconditional DiT models; set `model.conditioning` in the config to select
the mode.

```bash
python -m scripts.train_cifar10 \
    --config configs/cifar10_pixel.yaml \
    --output_dir ./outputs/cifar10_pixel
```

Use `configs/cifar10_pixel_test.yaml` for a two-step smoke run. To generate
class-conditional samples from an EMA checkpoint:

```bash
python -m scripts.sample_cifar10 \
    --config configs/cifar10_pixel.yaml \
    --checkpoint ./outputs/cifar10_pixel/checkpoints/5000 \
    --class_id 3 \
    --output_path ./samples/cifar10.png
```

Pixel-space sampling clips predicted clean images to `[-1, 1]`; the existing
latent-VAE sampling workflows retain their previous behavior.

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

To generate samples for one Fashion MNIST class, use the dedicated
class-conditional script. Fashion MNIST class IDs are 0–9.

```bash
python -m scripts.sample_fashion_mnist_conditional \
    --config configs/fashion_mnist.yaml \
    --checkpoint ./outputs/fashion_mnist/checkpoints/1000 \
    --class_id 7 \
    --num_samples 16 \
    --output_path ./samples/fashion_mnist_sneakers.png
```

The `--checkpoint` argument also accepts an Orbax checkpoint stored in GCS,
for example `gs://diffjax/models/fashion-mnist_tpu_09-08-2026/checkpoints/12000`.
The checkpoint is downloaded to `~/.cache/ldmax/checkpoints` and reused on
subsequent runs. Configure Google Cloud Application Default Credentials before
sampling from a private bucket.

To continue Fashion MNIST training from the latest checkpoint in an existing
run, provide `--resume_from` and a new `--output_dir`. `training.total_steps`
is the absolute target step, so this resumes step 5000 toward step 10000:

```bash
python -m scripts.train_fashion_mnist \
    --config configs/fashion_mnist.yaml \
    --resume_from models/fashion-mnist_ccond_cpu_10-08-2026 \
    --output_dir models/fashion-mnist_ccond_cpu_10-08-2026_resume
```

New checkpoints include the RNG state. Older checkpoints without RNG state,
including the step-5000 checkpoint above, use a deterministic seed-and-step
fallback, so their continuation cannot reproduce the exact original random
stream after the saved step.

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
