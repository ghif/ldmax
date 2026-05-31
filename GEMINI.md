# ldmax Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-05-29

## Active Technologies

- Python 3.10+ (standard for JAX/Flax NNX) + JAX, Flax NNX, Grain, Optax, Orbax (checkpoints), TensorBoard, Hugging Face Hub (001-jax-dit-training)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.10+ (standard for JAX/Flax NNX): Follow standard conventions

## Recent Changes

- 001-jax-dit-training: Added Python 3.10+ (standard for JAX/Flax NNX) + JAX, Flax NNX, Grain, Optax, Orbax (checkpoints), TensorBoard, Hugging Face Hub
- U-Net Implementation: Added Rombach 2022 U-Net architecture as an alternative to DiT, with support for class and attribute conditioning.

<!-- MANUAL ADDITIONS START -->
- Always run this code in conda environment `jax-cpu` or `jax-tpu`
<!-- MANUAL ADDITIONS END -->
