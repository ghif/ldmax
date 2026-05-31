<!--
SYNC IMPACT REPORT
- Version change: Template → 1.0.0
- Principles established: 
  1. Educational First & Clear Documentation
  2. Clean Code Architecture
  3. Collaborative AI Research
  4. Visual Generative Focus
  5. Deterministic Reproducibility
- Sections updated: Core Principles, Development Standards, Code Review Process, Governance.
- Templates checked: 
  - .specify/templates/plan-template.md (Updated Gates reference)
  - .specify/templates/spec-template.md (Verified)
  - .specify/templates/tasks-template.md (Verified)
- Follow-up TODOs: None.
-->

# LDMAX Constitution

## Core Principles

### I. Educational First & Clear Documentation
Code MUST be exceptionally well-documented (including inline comments, architectural guides, and comprehensive docstrings). Documentation and readability take precedence over extreme performance optimizations where the two conflict.
**Rationale**: Facilitates seamless onboarding for students and researchers, ensuring the codebase serves as a learning resource.

### II. Clean Code Architecture
The repository MUST adhere to clean code principles, including SOLID design patterns, clear separation of concerns, and robust abstraction layers. Use dependency injection for model components to ensure testability.
**Rationale**: Maintains a highly maintainable and robust structure as research complexity grows.

### III. Collaborative AI Research
Architecture MUST be highly modular, enabling researchers to easily swap model components, datasets, and training loop implementations without modifying core framework logic.
**Rationale**: Encourages diverse experimentation and contributions from a broad research community.

### IV. Visual Generative Focus
System tooling, data pipelines, and performance optimizations MUST prioritize spatial data processing, as well as image and video generation tasks.
**Rationale**: Aligns repository development with the primary research domain of visual generative models.

### V. Deterministic Reproducibility (NON-NEGOTIABLE)
All experiments and generative pipelines MUST enforce fixed random seeds by default and specify exact environment constraints (versions, hardware requirements).
**Rationale**: Reproducibility is the cornerstone of credible AI research and collaboration.

## Development Standards
All new code must undergo automated linting and pass a suite of unit tests that verify both behavioral correctness and adherence to the clean architecture principles.

## Code Review Process
Pull Requests must be reviewed for both technical accuracy and "educational clarity." A PR that is functionally correct but architecturally opaque should be refactored for clarity.

## Governance
The LDMAX Constitution is the foundational document for project governance. Any amendments to these principles require a version bump and updates to all dependent templates.

**Version**: 1.0.0 | **Ratified**: 2026-05-29 | **Last Amended**: 2026-05-29
