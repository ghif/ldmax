"""Main training script for DiT."""

import os
import time
from absl import app, flags
from flax import nnx
import optax
import jax
import jax.numpy as jnp
from tqdm import tqdm

from src.models.dit.dit import DiT
from src.data.cifar import get_cifar10_dataset
from src.data.celeba import get_celeba_dataset
from src.training.step import train_step
from src.utils.rng import RNGManager
from src.utils.config import load_config
from src.utils.logging import TensorBoardLogger
from src.utils.checkpoint import CheckpointManager

from src.training.sampler import DDIMSampler
from src.utils.vae import VAEManager
from src.training.ema import EMAManager

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/cifar10.yaml", "Path to the config file.")
flags.DEFINE_string("output_dir", "./outputs", "Output directory.")

def main(_):
    # 1. Load config
    config = load_config(FLAGS.config)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    # 2. Setup RNG, Logger, Checkpointer, VAE, Sampler
    rng_manager = RNGManager(config.training.seed)
    logger = TensorBoardLogger(os.path.join(FLAGS.output_dir, "logs"))
    checkpointer = CheckpointManager(os.path.join(FLAGS.output_dir, "checkpoints"))
    sampler = DDIMSampler()
    vae_manager = VAEManager()
    
    # Jitted VAE encoding
    @jax.jit
    def encode_fn(images, key):
        return vae_manager.encode(images, key)

    # 3. Initialize Model and Optimizer
    model = DiT(
        input_size=config.model.input_size,
        patch_size=config.model.patch_size,
        in_channels=config.model.in_channels,
        hidden_size=config.model.hidden_size,
        depth=config.model.depth,
        num_heads=config.model.num_heads,
        num_classes=config.model.num_classes,
        rngs=nnx.Rngs(rng_manager.next())
    )
    
    # Optional EMA
    ema = EMAManager(model, decay=config.training.ema_decay) if hasattr(config.training, "ema_decay") else None
    
    optimizer = nnx.Optimizer(
        model, 
        optax.adamw(config.training.learning_rate, weight_decay=config.training.weight_decay),
        wrt=nnx.Param
    )
    
    # 4. Load Data
    if config.dataset == "cifar10":
        dataset = get_cifar10_dataset(
            batch_size=config.training.batch_size,
            shuffle=True,
            seed=config.training.seed
        )
    elif config.dataset == "celeba":
        dataset = get_celeba_dataset(
            batch_size=config.training.batch_size,
            shuffle=True,
            seed=config.training.seed,
            target_size=config.data.image_size
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # 5. Training Loop
    print(f"Starting training for {config.training.total_steps} steps...")
    use_bf16 = getattr(config.training, "mixed_precision", False)
    
    # Simple training loop
    for step in tqdm(range(config.training.total_steps)):
        batch = next(iter(dataset))
        
        # 1. VAE Encoding (latents)
        latents = encode_fn(batch["image"], rng_manager.next())
        
        # 2. Training Step
        metrics = train_step(
            model, 
            optimizer, 
            latents,
            batch["label"],
            rng_manager.next(), 
            use_bf16=use_bf16
        )
        
        if ema is not None:
            ema.update(model)
        
        # Logging
        if step % config.evaluation.log_interval == 0:
            logger.log_scalars(step, {"train/loss": float(metrics["loss"])})
            
        # Sampling
        if step % config.evaluation.sampling_interval == 0:
            sample_labels = jnp.zeros((config.training.batch_size,), dtype=jnp.int32)
            
            # create a temporary model for sampling to apply EMA without affecting training weights
            sampling_model = DiT(
                input_size=config.model.input_size,
                patch_size=config.model.patch_size,
                in_channels=config.model.in_channels,
                hidden_size=config.model.hidden_size,
                depth=config.model.depth,
                num_heads=config.model.num_heads,
                num_classes=config.model.num_classes,
                rngs=nnx.Rngs(rng_manager.next())
            )
            
            if ema is not None:
                ema.apply_to(sampling_model)
            else:
                nnx.update(sampling_model, nnx.state(model))
            
            @nnx.jit
            def model_fn(x, t, y):
                out = sampling_model(x, t, y)
                if out.shape[-1] == x.shape[-1] * 2:
                    return jnp.split(out, 2, axis=-1)[0]
                return out
                
            sample_shape = (len(sample_labels), config.model.input_size, config.model.input_size, config.model.in_channels)
            samples = sampler.sample(model_fn, sample_shape, rng_manager.next(), y=sample_labels)
            
            # Decode samples back to pixel space for visualization
            samples_pixel = vae_manager.decode(samples)
            logger.log_images(step, "train/samples", samples_pixel)

        # Checkpointing
        if (step % config.evaluation.checkpoint_interval == 0 and step > 0) or (step == config.training.total_steps - 1):
            checkpointer.save(step, {
                "model": nnx.state(model),
                "ema": ema.state if ema is not None else None,
                "opt": nnx.state(optimizer),
                "step": step
            })
            
    logger.close()
    print("Training complete.")

if __name__ == "__main__":
    app.run(main)
