# Tasks: JAX DiT Training Pipeline

**Input**: Design documents from `/specs/001-jax-dit-training/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Test tasks are included below as verification steps are critical for research reproducibility.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project structure (src/models, src/data, src/training, src/utils, tests, configs, scripts) per refactored plan.md
- [X] T002 Initialize Python project with JAX, Flax NNX, Grain, Optax, Orbax dependencies
- [X] T003 [P] Configure linting (Ruff) and formatting tools per GEMINI.md

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T004 Implement centralized PRNGKey management in src/utils/rng.py
- [X] T005 [P] Setup Orbax checkpointer configuration in src/utils/checkpoint.py
- [X] T006 [P] Implement YAML configuration parsing in src/utils/config.py
- [X] T007 Implement basic DiT block components in src/models/dit/blocks.py
- [X] T008 [P] Setup TensorBoard logging utilities in src/utils/logging.py

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Model Training Initialization (Priority: P1) 🎯 MVP

**Goal**: Initialize a Diffusion Transformer (DiT) training run from scratch on CIFAR-10.

**Independent Test**: Run `python -m src.scripts.train --config configs/test_cifar10.yaml` and verify first 10 steps complete.

### Tests for User Story 1

- [X] T009 [P] [US1] Unit test for DiT model initialization in tests/unit/test_model.py
- [X] T010 [P] [US1] Unit and visualization tests for Grain data pipeline in tests/unit/test_cifar.py

### Implementation for User Story 1

- [X] T011 [P] [US1] Implement DiT model using Flax NNX in src/models/dit/dit.py
- [X] T012 [P] [US1] Implement Grain data pipeline for CIFAR-10 in src/data/cifar.py
- [X] T013 [US1] Implement basic training step with Optax in src/training/step.py
- [X] T014 [US1] Create main training entry point script src/scripts/train.py
- [X] T015 [US1] Implement VAE auto-download and loading in src/utils/vae.py

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Training Progress Monitoring & Sampling (Priority: P2)

**Goal**: Monitor training via loss metrics and periodic image sampling logged to TensorBoard.

**Independent Test**: Verify TensorBoard dashboard displays loss curves and sampled image grids.

### Tests for User Story 2

- [X] T016 [P] [US2] Unit test for image sampling logic in tests/unit/test_sampler.py

### Implementation for User Story 2

- [X] T017 [US2] Implement diffusion sampling loop in src/training/sampler.py
- [X] T018 [US2] Integrate image sampling into training loop in src/scripts/train.py
- [X] T019 [P] [US2] Implement quantitative metrics (FID) using jax-fid in src/utils/metrics.py

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Experiment Configuration & Reproducibility (Priority: P3)

**Goal**: Support EMA weights, mixed-precision (BF16), and full architecture configuration.

**Independent Test**: Verify identical results from two runs with the same seed and BF16 enabled.

### Tests for User Story 3

- [X] T020 [P] [US3] Unit test for EMA weight update in tests/unit/test_ema.py
- [X] T021 [P] [US3] Unit test for BF16 mixed-precision stability in tests/unit/test_precision.py

### Implementation for User Story 3

- [X] T022 [US3] Implement EMA weight tracking in src/training/ema.py
- [X] T023 [US3] Integrate bfloat16 mixed-precision in training loop src/training/step.py
- [X] T024 [US3] Create standalone inference script src/scripts/sample.py
- [X] T025 [P] [US3] Create baseline YAML configurations in configs/cifar10.yaml

**Checkpoint**: All user stories should now be independently functional

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T026 [P] Add comprehensive docstrings and architectural notes to all modules
- [X] T027 Final integration test for full pipeline in tests/integration/test_pipeline.py
- [X] T028 [P] Run and validate quickstart.md instructions
