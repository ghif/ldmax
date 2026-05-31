# Feature Specification: JAX DiT Training Pipeline

**Feature Branch**: `001-jax-dit-training`  
**Created**: 2026-05-29  
**Status**: Draft  
**Input**: User description: "as a researcher, I want to comfortably train a latent diffusion model with a full transformer architecture (DiT, Peeble et al. 2023) from scratch with JAX implementation (Grain data processing, Flax NNX, Optax)"

## Clarifications

### Session 2026-05-29
- Q: Is multi-node (distributed) training required for the initial implementation, or should we focus on single-node (multi-accelerator) efficiency first? → A: Single-node only (multi-GPU/TPU)
- Q: Should the system include a mechanism to automatically download pre-trained VAE weights from a public repository, or should it strictly expect a local file path? → A: Auto-download from Hub (e.g., HF)
- Q: Which primary logging and monitoring framework should be integrated as the default for tracking training metrics and visual samples? → A: TensorBoard
- Q: Should the Grain data pipeline support on-the-fly VAE encoding or strictly expect a pre-computed latent dataset? → A: Both (On-the-fly and Pre-computed)
- Q: Should the training pipeline support mixed-precision training (e.g., bfloat16) by default? → A: bfloat16 (Recommended for performance)
- Q: For the initial implementation starting with CIFAR, should the system prioritize built-in support for standard research datasets or focus on a flexible local directory loader? → A: Built-in CIFAR (HF/TFDS) + Local Folder, using Grain loader.
- Q: Should the initial DiT implementation support class-conditional training or start with unconditional generation? → A: Class-conditional (using labels)
- Q: Should the training pipeline maintain and checkpoint an Exponential Moving Average (EMA) of the model weights? → A: Implement and checkpoint EMA weights
- Q: Should the implementation include a standalone inference script for generating images from saved checkpoints? → A: Standalone inference script required
- Q: Should the system include automated calculation of standard generative metrics like FID during the evaluation/sampling phase? → A: Implement automated FID calculation

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Model Training Initialization (Priority: P1)

As a researcher, I want to initialize a Diffusion Transformer (DiT) training run from scratch, so that I can begin my research experiments with minimal setup friction.

**Why this priority**: Foundational requirement. Without initialization and a basic training loop, no research can proceed.

**Independent Test**: Can be tested by running the main training script with a small configuration and verifying that the model initializes, the data loader starts, and the first training steps complete without error.

**Acceptance Scenarios**:

1. **Given** a correctly configured environment, **When** the training script is executed with default parameters, **Then** the transformer model should be instantiated correctly and training steps should begin.
2. **Given** a target dataset path, **When** the script starts, **Then** the data pipeline should successfully initialize and feed batches to the model.

---

### User Story 2 - Training Progress Monitoring & Sampling (Priority: P2)

As a researcher, I want to monitor the training progress via loss metrics and periodic image sampling, so that I can evaluate the model's learning progress and visual quality.

**Why this priority**: Essential for evaluating model convergence and output quality during long-running experiments.

**Independent Test**: Verify that TensorBoard logs show decreasing loss and that generated images are visible in the TensorBoard dashboard or output directory at specified intervals.

**Acceptance Scenarios**:

1. **Given** an active training run, **When** a specified step interval is reached, **Then** the system MUST generate sample images from the current model state and log them to TensorBoard.
2. **Given** an active training run, **When** metrics are calculated, **Then** loss and other diagnostic values MUST be logged to TensorBoard format.

---

### User Story 3 - Experiment Configuration & Reproducibility (Priority: P3)

As a researcher, I want to easily configure model architecture parameters and training settings via a configuration file, so that I can systematically explore different model configurations and reproduce results.

**Why this priority**: Facilitates structured research and aligns with the project's core principles of reproducibility.

**Independent Test**: Run two separate training sessions with different configuration files and verify that the model architecture and training hyperparameters change accordingly.

**Acceptance Scenarios**:

1. **Given** a configuration file with modified model dimensions, **When** training starts, **Then** the instantiated model MUST reflect these dimensions.
2. **Given** a fixed random seed in the configuration, **When** running identical experiments, **Then** the initial weights and first batch samples MUST be identical.

### Edge Cases

- **Checkpoint Corruption**: If a training run is interrupted during a checkpoint write, the system SHOULD maintain the previous valid checkpoint to prevent total loss of progress.
- **Out of Memory (OOM)**: The system SHOULD provide clear guidance or error messages when the requested model size or batch size exceeds available accelerator memory.
- **Empty/Invalid Dataset**: If the dataset directory is empty or contains incompatible files, the system MUST exit gracefully with a descriptive error message.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST implement a transformer-based neural network architecture optimized for class-conditional diffusion tasks (specifically DiT).
- **FR-002**: System MUST utilize a high-performance Grain-based data loading pipeline supporting both standard research datasets (e.g., CIFAR via HF/TFDS) and local directory loaders.
- **FR-003**: System MUST implement a training loop that efficiently utilizes hardware accelerators (GPUs/TPUs) in a single-node configuration.
- **FR-004**: System MUST support configurable gradient optimization strategies (e.g., weight decay, learning rate scheduling).
- **FR-005**: System MUST provide a mechanism for periodic checkpointing of both model parameters (including EMA weights) and optimizer state.
- **FR-006**: System MUST allow full configuration of model dimensions (e.g., depth, width, attention heads) without modifying source code.
- **FR-007**: System MUST support training on latent representations, supporting both on-the-fly VAE encoding of raw images and direct loading of pre-computed latent datasets. It MUST include a mechanism to automatically retrieve pre-trained VAE weights from external repositories.
- **FR-008**: System MUST support mixed-precision training (specifically `bfloat16`) to optimize throughput and memory usage on compatible accelerators.
- **FR-009**: System MUST provide a standalone inference script to generate images from saved model checkpoints (specifically using EMA weights if available).
- **FR-010**: System MUST support automated calculation of quantitative metrics (specifically FID) during the periodic evaluation phase.

### Implementation Constraints

- **IC-001**: Implementation MUST be written in JAX.
- **IC-002**: Model components MUST use the Flax NNX library.
- **IC-003**: Optimization and gradient transformations MUST use the Optax library.
- **IC-004**: Data processing and batching pipelines MUST use the Grain library.

### Key Entities

- **Model**: The transformer-based neural network.
- **Data Pipeline**: The pipeline responsible for fetching, augmenting, and preparing training data.
- **Trainer**: The orchestrator managing the training loop and state persistence.
- **Configuration**: The definition of hyperparameters and environment settings.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A researcher can initiate a training run with a single command-line execution after environment setup.
- **SC-002**: Training throughput (samples/sec) remains consistent within 10% variance after the initial 100 steps.
- **SC-003**: Model checkpoints can be reloaded to resume training with bit-accurate continuity for deterministic operations.
- **SC-004**: Generated samples are produced and saved within 60 seconds of hitting the sampling interval on target hardware.

## Assumptions

- **A-001**: A pre-trained autoencoder (VAE) is available for converting images to latent representations.
- **A-002**: The researcher has access to JAX-compatible accelerators with sufficient memory for the requested model size.
- **A-003**: The initial implementation targets the CIFAR dataset. Training data follows either a standard format (e.g., directory of images) or is accessible via supported dataset libraries (HF/TFDS).
