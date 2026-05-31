# Quickstart: JAX DiT Training

This guide covers setting up and running the Diffusion Transformer (DiT) training pipeline on CIFAR-10.

## 1. Prerequisites
Ensure you have a JAX-compatible environment (GPU or TPU recommended).
```bash
pip install -r requirements.txt
```

## 2. Configuration
Create a configuration file `configs/cifar10.yaml`:
```yaml
model:
  depth: 12
  hidden_size: 384
  num_heads: 6
  patch_size: 2
training:
  learning_rate: 0.0001
  batch_size: 128
  total_steps: 100000
  ema_decay: 0.999
  mixed_precision: true
evaluation:
  sampling_interval: 1000
  fid_interval: 10000
```

## 3. Start Training
Launch the training script. The system will automatically download the required VAE weights from Hugging Face if not present.
```bash
python -m src.scripts.train --config configs/cifar10.yaml
```

## 4. Monitor Progress
View training metrics and periodic samples in TensorBoard:
```bash
tensorboard --logdir ./outputs/logs
```

## 5. Generate Samples
Use the standalone inference script to generate a grid of images from a saved checkpoint:
```bash
python -m src.scripts.sample --checkpoint ./outputs/checkpoints/step_100000 --num_samples 64
```
