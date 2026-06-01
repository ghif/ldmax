"""Main training script for DiT."""

import os
import time
from absl import app, flags
from flax import nnx
import optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from tqdm import tqdm

from src.models.dit.dit import DiT
from src.models.unet.unet import UNetModel
from src.data.cifar import get_cifar10_dataset
from src.data.celeba import get_celeba_dataset
from src.training.step import train_step
from src.utils.rng import RNGManager
from src.utils.config import load_config
from src.utils.logging import TensorBoardLogger
from src.utils.checkpoint import CheckpointManager
from src.utils.prefetch import DevicePrefetcher

from src.training.sampler import DDIMSampler
from src.utils.vae import VAEManager
from src.training.ema import EMAManager

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/cifar10.yaml", "Path to the config file.")
flags.DEFINE_string("output_dir", "./outputs", "Output directory.")

def main(_):
    # 0. Setup Mesh for Multi-Device (TPU/CPU fallback)
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))
    data_sharding = NamedSharding(mesh, P('data'))
    replicate_sharding = NamedSharding(mesh, P())
    print(f"Using {len(devices)} devices: {devices}")

    with jax.set_mesh(mesh):
        # 1. Load config
        config = load_config(FLAGS.config)
        os.makedirs(FLAGS.output_dir, exist_ok=True)
        
        use_bf16 = config.training.get("mixed_precision", False)
        
        # 2. Setup RNG, Logger, Checkpointer, VAE, Sampler
        rng_manager = RNGManager(config.training.seed)
        logger = TensorBoardLogger(os.path.join(FLAGS.output_dir, "logs"))
        checkpointer = CheckpointManager(os.path.join(FLAGS.output_dir, "checkpoints"))
        sampler = DDIMSampler()
        vae_manager = VAEManager()
        
        # Replicate VAE parameters across all devices
        vae_manager.params = jax.device_put(vae_manager.params, replicate_sharding)
        
        # Jitted VAE encoding
        @jax.jit
        def encode_fn(images, key):
            return vae_manager.encode(images, key)

        # 3. Initialize Model and Optimizer
        model_type = config.model.get("type", "dit")
        if model_type == "dit":
            model = DiT(
                input_size=config.model.input_size,
                patch_size=config.model.patch_size,
                in_channels=config.model.in_channels,
                hidden_size=config.model.hidden_size,
                depth=config.model.depth,
                num_heads=config.model.num_heads,
                num_classes=config.model.num_classes,
                label_mode=getattr(config.model, "label_mode", "class"),
                label_dim=getattr(config.model, "label_dim", None),
                learn_sigma=config.model.get("learn_sigma", False),
                rngs=nnx.Rngs(rng_manager.next())
            )
        elif model_type == "unet":
            model = UNetModel(
                in_channels=config.model.in_channels,
                out_channels=config.model.get("out_channels", config.model.in_channels),
                model_channels=config.model.model_channels,
                attention_resolutions=config.model.attention_resolutions,
                num_res_blocks=config.model.num_res_blocks,
                channel_mult=config.model.channel_mult,
                num_heads=config.model.num_heads,
                transformer_depth=config.model.get("transformer_depth", 1),
                context_dim=config.model.get("context_dim", None),
                num_classes=config.model.num_classes,
                label_mode=getattr(config.model, "label_mode", "class"),
                label_dim=getattr(config.model, "label_dim", None),
                dropout=config.training.get("dropout", 0.0),
                rngs=nnx.Rngs(rng_manager.next())
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Replicate model state
        model_state = nnx.state(model)
        if use_bf16:
            model_state = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, model_state)
        nnx.update(model, jax.device_put(model_state, replicate_sharding))
        
        # Optional EMA
        ema = EMAManager(model, decay=config.training.ema_decay) if hasattr(config.training, "ema_decay") else None
        if ema is not None:
             # EMA state is a state object, needs replication
             ema.ema_state = jax.device_put(ema.ema_state, replicate_sharding)
        
        optimizer = nnx.Optimizer(
            model, 
            optax.adamw(config.training.learning_rate, weight_decay=config.training.weight_decay),
            wrt=nnx.Param
        )
        
        if use_bf16:
             # Ensure optimizer state is sharded and ideally in fp32 for stability 
             # (Optax does this by default if we don't cast it)
             pass

        # Replicate optimizer state
        nnx.update(optimizer, jax.device_put(nnx.state(optimizer), replicate_sharding))
        
        # 3b. Setup Sampling Model and JITted function (outside loop)
        if model_type == "dit":
            sampling_model = DiT(
                input_size=config.model.input_size,
                patch_size=config.model.patch_size,
                in_channels=config.model.in_channels,
                hidden_size=config.model.hidden_size,
                depth=config.model.depth,
                num_heads=config.model.num_heads,
                num_classes=config.model.num_classes,
                label_mode=getattr(config.model, "label_mode", "class"),
                label_dim=getattr(config.model, "label_dim", None),
                learn_sigma=config.model.get("learn_sigma", False),
                rngs=nnx.Rngs(rng_manager.next())
            )
        else:
            sampling_model = UNetModel(
                in_channels=config.model.in_channels,
                out_channels=config.model.get("out_channels", config.model.in_channels),
                model_channels=config.model.model_channels,
                attention_resolutions=config.model.attention_resolutions,
                num_res_blocks=config.model.num_res_blocks,
                channel_mult=config.model.channel_mult,
                num_heads=config.model.num_heads,
                transformer_depth=config.model.get("transformer_depth", 1),
                context_dim=config.model.get("context_dim", None),
                num_classes=config.model.num_classes,
                label_mode=getattr(config.model, "label_mode", "class"),
                label_dim=getattr(config.model, "label_dim", None),
                dropout=config.training.get("dropout", 0.0),
                rngs=nnx.Rngs(rng_manager.next())
            )
        # Ensure sampling model is on the mesh
        nnx.update(sampling_model, jax.device_put(nnx.state(sampling_model), replicate_sharding))

        @nnx.jit
        def model_fn(model, x, t, y):
            out = model(x, t, y)
            # If learn_sigma is True, model_output has 2*C channels. 
            # We take the first C channels for noise prediction.
            if out.shape[-1] == x.shape[-1] * 2:
                return jnp.split(out, 2, axis=-1)[0]
            return out

        # 4. Load Data
        if config.dataset == "cifar10":
            dataset = get_cifar10_dataset(
                batch_size=config.training.batch_size,
                shuffle=True,
                seed=config.training.seed,
                target_size=getattr(config.data, "image_size", None)
            )
        elif config.dataset == "celeba":
            dataset = get_celeba_dataset(
                batch_size=config.training.batch_size,
                shuffle=True,
                seed=config.training.seed,
                target_size=config.data.image_size,
                dataset_name=getattr(config.data, "dataset_name", "flwrlabs/celeba"),
                dataset_config=getattr(config.data, "dataset_config", "img_align+identity+attr")
            )
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")
        
        # Wrap dataset with device prefetcher
        dataset = DevicePrefetcher(dataset, data_sharding)
        dataset_iter = iter(dataset)

        # 5. Training Loop
        print(f"Starting training for {config.training.total_steps} steps...")
        
        # Simple training loop
        for step in tqdm(range(config.training.total_steps)):
            batch = next(dataset_iter)
            
            # Batch is already sharded and on device thanks to DevicePrefetcher
            batch_images = batch["image"]
            batch_labels = batch["label"]
            
            # 1. VAE Encoding (latents)
            latents = encode_fn(batch_images, rng_manager.next())
            
            # 2. Training Step
            metrics = train_step(
                model, 
                optimizer, 
                latents,
                batch_labels,
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
                sample_count = min(
                    getattr(config.evaluation, "sample_count", 16),
                    config.training.batch_size,
                )
                if config.dataset == "celeba":
                    # Use a few in-batch CelebA attribute vectors for visualization.
                    num_samples = sample_count
                    sample_labels = batch_labels[:num_samples]

                    sample_labels = jax.device_put(sample_labels, data_sharding)
                    null_labels = jnp.zeros_like(sample_labels)
                else:
                    # Use a mix of classes for visualization
                    num_samples = sample_count
                    sample_labels = jnp.arange(num_samples, dtype=jnp.int32) % max(1, config.model.num_classes)
                    
                    sample_labels = jax.device_put(sample_labels, data_sharding)
                    null_labels = jnp.full_like(sample_labels, fill_value=config.model.num_classes)
                
                # Update sampling model with EMA weights or current model weights
                if ema is not None:
                    ema.apply_to(sampling_model)
                else:
                    nnx.update(sampling_model, nnx.state(model))
                
                sample_shape = (len(sample_labels), config.model.input_size, config.model.input_size, config.model.in_channels)
                samples = sampler.sample(
                    lambda x, t, y: model_fn(sampling_model, x, t, y), 
                    sample_shape, 
                    rng_manager.next(), 
                    y=sample_labels, 
                    null_y=null_labels,
                    cfg_scale=4.0
                )
                
                # Decode samples back to pixel space for visualization
                samples_pixel = vae_manager.decode(samples)
                logger.log_images(step, "train/samples", samples_pixel[:num_samples])

            # Checkpointing
            if (step % config.evaluation.checkpoint_interval == 0 and step > 0) or (step == config.training.total_steps - 1):
                checkpointer.save(step, {
                    "model": nnx.state(model),
                    "ema": ema.ema_state if ema is not None else None,
                    "opt": nnx.state(optimizer),
                    "step": step
                })
            
    logger.close()
    print("Training complete.")

if __name__ == "__main__":
    app.run(main)
