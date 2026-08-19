# AGENTS.md — Agent Guidelines & Repository Guide

Welcome to **LDMAX (Latent Diffusion Models in JAX)**. This document serves as the primary technical guide for AI coding agents and autonomous workflows operating within this repository.

---

## ⚠️ Critical Operational Requirements

> [!IMPORTANT]
> **Environment Requirement**: All commands, test runs, and training/sampling scripts in this repository **MUST** be executed inside the **`ldmax`** conda environment.
>
> ```bash
> conda activate ldmax
> ```
>
> Always ensure `PYTHONPATH=.` or `PYTHONPATH=$PYTHONPATH:.` is set when running scripts from the project root.

---

## 🧠 Project Architecture & Ecosystem

LDMAX is an educational, modular, and performant codebase for training and sampling from **Diffusion Transformers (DiT)** from scratch in the modern JAX ecosystem.

### Key Technologies
- **Core Framework**: JAX (`jax`, `jax.numpy`), JIT compilation (`jax.jit`).
- **Model Definition**: **Flax NNX** (`flax.nnx`), the reference-based object-oriented API for Flax.
- **Optimization**: Optax (`optax`) for gradient transformations and AdamW.
- **Checkpointing**: Orbax (`orbax.checkpoint`) with async saving and GCS sync support.
- **Data Loading**: Google Grain (`grain.python`) for high-throughput sharded pipelines (CIFAR-10, CelebA), alongside Hugging Face `datasets` / NumPy in-memory iterators (Fashion-MNIST).
- **Precision**: Mixed-precision support with `bfloat16` and explicit FP32 parameter/schedule management for TPU/GPU execution.

---

## 📁 Repository Structure

```text
ldmax/
├── configs/               # YAML experiment configurations (dataset, model, training params)
│   ├── celeba*.yaml       # CelebA latent-space configurations
│   ├── cifar10*.yaml      # CIFAR-10 latent & native-pixel configurations
│   └── fashion_mnist*.yaml# Fashion-MNIST raw-pixel configurations
├── docs/                  # Specs, design documents, and research notes
├── scripts/               # CLI entry points (thin wrappers calling src/ workflows)
│   ├── train_cifar10.py   # Launch native-pixel CIFAR-10 training
│   ├── train_fashion_mnist.py # Launch raw-pixel Fashion-MNIST training
│   ├── train_celeba.py    # Launch CelebA latent diffusion training
│   ├── train.py           # Unified training entry point
│   ├── sample*.py         # Sampling / evaluation scripts
│   └── demo_*.py          # Interactive demos
├── src/                   # Core library code
│   ├── data/              # Dataset sources & pipelines (cifar.py, celeba.py, fashion_mnist.py)
│   ├── models/            # Neural network architectures
│   │   └── dit/           # Diffusion Transformer (DiT) & AdaLN-Zero blocks
│   ├── sampling/          # Sampling logic, class-conditional grids, HF DiT loading
│   ├── training/          # Training runners, step functions, EMA, DDIM sampler
│   └── utils/             # Checkpoint managers, configs, logging, prefetching, RNG, VAE
├── tests/                 # Unit & integration tests
│   ├── unit/              # Unit tests for models, data loaders, checkpointing, and runners
│   └── integration/       # End-to-end pipeline integration tests
├── tpu/                   # TPU VM orchestration, startup scripts, and benchmarks
├── requirements_cpu.txt   # CPU dependencies
├── requirements_gpu.txt   # GPU (CUDA) dependencies
├── requirements_tpu.txt   # TPU dependencies
└── pyproject.toml         # Ruff linting & formatting configurations
```

---

## 🛠️ Common Commands & Workflows

Ensure you are in the `ldmax` conda environment before running:

### 1. Running Tests

```bash
# Run unit tests on CPU
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/unit/

# Run specific unit test modules
JAX_PLATFORMS=cpu PYTHONPATH=. pytest tests/unit/test_cifar10_pixel.py tests/unit/test_model.py
```

### 2. Linting & Formatting

```bash
# Check lint rules
ruff check .

# Format code
ruff format .
```

### 3. Training Workflows

```bash
# CIFAR-10 Native-Pixel Diffusion
PYTHONPATH=. python scripts/train_cifar10.py \
    --config configs/cifar10_pixel.yaml \
    --output_dir models/cifar10_pixel_run

# Fashion-MNIST Raw-Pixel Diffusion
PYTHONPATH=. python scripts/train_fashion_mnist.py \
    --config configs/fashion_mnist.yaml \
    --output_dir models/fashion_mnist_run

# CelebA Latent Diffusion (using VAE)
PYTHONPATH=. python scripts/train_celeba.py \
    --config configs/celeba.yaml \
    --output_dir models/celeba_run
```

### 4. Sampling & Evaluation

```bash
# Sample class-conditional CIFAR-10 images from checkpoint
PYTHONPATH=. python scripts/sample_cifar10.py \
    --config configs/cifar10_pixel.yaml \
    --checkpoint models/cifar10_pixel_run/checkpoints/5000 \
    --class_id 3 \
    --output_path samples/cifar10_class3.png

# Sample Fashion-MNIST
PYTHONPATH=. python scripts/sample_fashion_mnist.py \
    --config configs/fashion_mnist.yaml \
    --checkpoint models/fashion_mnist_run/checkpoints/1000 \
    --output_path samples/fashion_mnist.png
```

---

## 🛡️ Coding Standards & Best Practices for Agents

1. **JAX Tracing & Multiprocessing Safety**:
   - Never evaluate `jnp` operations at module level (e.g. outside functions or classes). Evaluating `jnp` during module import causes spawned data-loader subprocesses (`grain` / `multiprocessing`) to attempt initializing hardware backends (TPU/GPU) concurrently, leading to process aborts.
   - Use standard NumPy (`np.linspace`, `np.array`) for module-level global definitions, and convert to JAX arrays (`jnp.asarray(...)`) inside JIT-compiled functions.
2. **Flax NNX State Management**:
   - Model modules inherit from `nnx.Module`.
   - RNG keys are handled via `nnx.Rngs` or explicit JAX PRNGKeys managed by `RNGManager`.
   - Optimizers are instantiated with `nnx.Optimizer(model, optax_tx, wrt=nnx.Param)`.
3. **Pure Functional Training Steps**:
   - Keep `compute_loss` and step functions stateless or explicitly decorated with `@nnx.jit`.
   - Ensure mixed-precision parameters stay in FP32 while intermediate matrix multiplications use BF16 when configured.
4. **Imports and Script Execution**:
   - CLI scripts under `scripts/` should remain thin dispatchers that parse flags and invoke runners under `src/training/` or `src/sampling/`.
   - Always maintain absolute package imports relative to `src` (e.g., `from src.models.dit.dit import DiT`).
