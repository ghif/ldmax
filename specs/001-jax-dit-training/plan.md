# Implementation Plan: JAX DiT Training Pipeline

**Branch**: `001-jax-dit-training` | **Date**: 2026-05-29 | **Spec**: [specs/001-jax-dit-training/spec.md](specs/001-jax-dit-training/spec.md)
**Input**: Feature specification from `/specs/001-jax-dit-training/spec.md`

## Summary

The objective is to build a high-performance training pipeline for Diffusion Transformers (DiT) using the JAX ecosystem. The system will leverage Flax NNX for model definition, Grain for data loading, and Optax for optimization. Key features include class-conditional training on CIFAR-10, EMA weight tracking, mixed-precision (bfloat16) support, and integration with TensorBoard for monitoring.

## Technical Context

**Language/Version**: Python 3.10+ (standard for JAX/Flax NNX)
**Primary Dependencies**: JAX, Flax NNX, Grain, Optax, Orbax (checkpoints), TensorBoard, Hugging Face Hub
**Storage**: Local filesystem for checkpoints and logs
**Testing**: pytest
**Target Platform**: Linux (single-node, GPU/TPU)
**Project Type**: Research CLI / Library
**Performance Goals**: Stable throughput (within 10% variance), fast sampling (SC-004)
**Constraints**: MUST use JAX, Flax NNX, Grain, and Optax. MUST support bfloat16.
**Scale/Scope**: Initial target: CIFAR-10 (32x32 images), single-node multi-accelerator training.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Educational First**: Code MUST be exceptionally well-documented. [Plan: Add docstring requirements to all modules]
- **Clean Code Architecture**: Adhere to SOLID and clear separation of concerns. [Plan: Use dependency injection for model/data/opt]
- **Collaborative AI Research**: Modular architecture for component swapping. [Plan: Define clear interfaces for DiT blocks and schedulers]
- **Visual Generative Focus**: Prioritize DiT and visual metrics (FID). [Plan: Integrated FID and image sampling]
- **Deterministic Reproducibility (NON-NEGOTIABLE)**: Enforce fixed random seeds. [Plan: Centralized PRNGKey management]

## Project Structure

### Documentation (this feature)

```text
specs/001-jax-dit-training/
├── plan.md              # This file
├── research.md          # Phase 0 output (COMPLETED)
├── data-model.md        # Phase 1 output (COMPLETED)
├── quickstart.md        # Phase 1 output (COMPLETED)
├── contracts/           # Phase 1 output (COMPLETED)
│   ├── train_cli.md
│   └── sample_cli.md
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/
├── models/
│   └── dit/             # Flax NNX DiT implementation (adaLN-Zero)
├── data/                # Grain pipelines (CIFAR-10, CelebA)
├── training/            # Training loop, optimization (BF16), and EMA
├── utils/               # Orbax, FID calculation (jax-fid), VAE, and TensorBoard
├── scripts/
│   ├── train.py         # Main training entry point
│   └── sample.py        # Standalone inference script
└── configs/             # YAML configurations (CIFAR-10)

tests/
├── integration/         # Full pipeline tests
└── unit/                # Component tests (Transformer, Grain, VAE)
```

**Structure Decision**: Architecture-agnostic modular structure. Core pipelines (Data, Training, Utils) are shared, while specific architectures are isolated in `src/models/`. Each module includes extensive docstrings and architectural notes (Educational First).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Hybrid Latent Strategy | Comfort/Performance tradeoff | Pre-computed only is rigid; On-the-fly only is slow for large scale. |
