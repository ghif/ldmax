# Inference CLI Contract: `sample.py`

## Command Usage
```bash
python -m src.scripts.sample --checkpoint <path_to_checkpoint> --num_samples <int> [OPTIONS]
```

## Required Arguments
- `--checkpoint` (str): Path to the saved model checkpoint (typically the Orbax directory).

## Optional Arguments
- `--num_samples` (int, default=16): Number of images to generate.
- `--batch_size` (int, default=8): Batch size for inference.
- `--class_label` (int, optional): The class label to condition on (0-9 for CIFAR-10). If omitted, samples from all classes or follows a specific distribution.
- `--cfg_scale` (float, default=1.5): Classifier-Free Guidance scale.
- `--num_steps` (int, default=50): Number of sampling steps (e.g., using DDIM or DPM-Solver).
- `--output_path` (str, default="./samples.png"): File path or directory to save generated images.
- `--use_ema` (bool, default=True): Whether to use the EMA weights from the checkpoint.

## Expected Behavior
1. **Load**: Initialize JAX and load the DiT model parameters from the checkpoint.
2. **VAE**: Load the pre-trained VAE decoder.
3. **Sample**: Perform the diffusion sampling process in the latent space.
4. **Decode**: Convert latents to pixel space using the VAE.
5. **Save**: Save a grid of generated images to the specified path.
