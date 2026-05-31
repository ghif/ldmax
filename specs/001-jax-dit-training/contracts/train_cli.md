# Training CLI Contract: `train.py`

## Command Usage
```bash
python -m src.scripts.train --config <path_to_yaml> [OVERRIDE_ARGS]
```

## Required Arguments
- `--config` (str): Path to the YAML configuration file defining model and training hyperparameters.

## Supported Overrides (Examples)
- `training.learning_rate`: Override the learning rate from the config.
- `training.batch_size`: Override the batch size.
- `model.depth`: Override the number of transformer layers.
- `output_dir`: Path where checkpoints and logs will be saved.

## Expected Behavior
1. **Setup**: Initialize JAX (detect GPU/TPU), setup logging, and load the pre-trained VAE.
2. **Data**: Initialize the Grain pipeline for CIFAR-10.
3. **Model**: Instantiate the DiT model with Flax NNX.
4. **Resumption**: If `output_dir` contains a valid checkpoint, automatically resume from the latest step.
5. **Loop**: Run the training loop, logging loss to TensorBoard and performing periodic sampling/checkpointing.
6. **Finalization**: Save the final model state and export the EMA weights.

## Output Structure
```text
output_dir/
├── checkpoints/         # Orbax checkpointer directory
├── logs/                # TensorBoard event files
└── samples/             # Periodically generated images (.png)
```
