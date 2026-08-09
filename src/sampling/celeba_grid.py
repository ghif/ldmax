"""Script to generate a grid of CelebA samples with varying attributes on the same latents."""

import os
import math
from absl import flags
from flax import nnx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import matplotlib.pyplot as plt

from src.models.dit.dit import DiT
from src.models.unet.unet import UNetModel
from src.training.sampler import DDIMSampler
from src.utils.checkpoint import CheckpointManager
from src.utils.vae import VAEManager
from src.utils.rng import RNGManager
from src.utils.config import load_config
from src.data.celeba import CELEBA_ATTRIBUTE_NAMES

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/celeba_tpu_b256.yaml", "Path to the config file.")
flags.DEFINE_string("checkpoint", "models/celeba_tpu_b256_opt/checkpoints/499999", "Path to the checkpoint.")
flags.DEFINE_integer("num_identities", 4, "Number of base identities (rows).")
flags.DEFINE_integer("num_steps", 50, "Number of sampling steps.")
flags.DEFINE_float("cfg_scale", 4.0, "Classifier-Free Guidance scale.")
flags.DEFINE_string("output_path", "./celeba_attribute_grid.png", "Output file path.")
flags.DEFINE_string(
    "attributes",
    "Smiling,Male,Blond_Hair,Eyeglasses,Young",
    "Comma-separated list of attributes to test as columns.",
)
flags.DEFINE_boolean("cpu_only", False, "Force CPU execution (useful for testing TPU checkpoints on CPU).")

def main(_):
    if FLAGS.cpu_only:
        jax.config.update("jax_platform_name", "cpu")
        print("Forced CPU execution via --cpu_only flag.")

    # 0. Setup Mesh (handles sharded checkpoints on CPU/TPU)
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))
    replicate_sharding = NamedSharding(mesh, P())
    
    with jax.set_mesh(mesh):
        # 1. Setup RNG
        rng_manager = RNGManager(42)
        base_key = rng_manager.next()

        # 2. Load config and build model
        config = load_config(FLAGS.config)
        label_dim = getattr(config.model, "label_dim", 40)
        label_mode = getattr(config.model, "label_mode", "attributes")
        
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
                label_mode=label_mode,
                label_dim=label_dim,
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
                label_mode=label_mode,
                label_dim=label_dim,
                rngs=nnx.Rngs(rng_manager.next())
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 3. Load Checkpoint
        if os.path.exists(FLAGS.checkpoint):
            print(f"Restoring from {FLAGS.checkpoint}...")
            
            import orbax.checkpoint as ocp
            
            # Paths MUST be absolute for many Orbax handlers (especially OCDBT)
            checkpoint_root = os.path.abspath(os.path.dirname(FLAGS.checkpoint))
            step_str = os.path.basename(FLAGS.checkpoint)
            try:
                step = int(step_str)
            except ValueError:
                step = None
                
            # Use a fresh CheckpointManager with StandardCheckpointer
            checkpointer = ocp.CheckpointManager(
                checkpoint_root, 
                {'default': ocp.StandardCheckpointer()}
            )
            
            # Recursive function to fix string keys from Orbax serialization
            def fix_keys(d):
                if isinstance(d, dict):
                    return {int(k) if k.isdigit() else k: fix_keys(v) for k, v in d.items()}
                return d

            try:
                # Use StandardRestore with fallback_sharding to force everything onto CPU.
                # By setting item=None, we avoid structural matching issues.
                # fallback_sharding ensures sharded arrays are loaded onto CPU as replicated.
                state = checkpointer.restore(
                    step, 
                    args=ocp.args.Composite(
                        default=ocp.args.StandardRestore(
                            item=None,
                            fallback_sharding=replicate_sharding
                        )
                    )
                )
                
                # Checkpointer returns a CompositeArgs/dict where 'default' is the actual state
                if hasattr(state, "default"):
                    state = state.default
                elif isinstance(state, dict) and "default" in state:
                    state = state["default"]
                
                if hasattr(state, "item"):
                    state = state.item
                
                # Preferred weights: EMA
                if state is not None and (isinstance(state, dict) or hasattr(state, "get")):
                    ckpt_weights = fix_keys(state.get("ema") or state.get("model"))
                    if ckpt_weights is not None:
                        print(f"Using {'EMA' if state.get('ema') is not None else 'model'} weights.")
                        
                        # Update model variables in-place to avoid replacing Variable objects
                        # which can lead to 'KeyError: Ellipsis' in NNX layers.
                        model_state = nnx.state(model)
                        
                        # State is a Mapping[tuple, Variable]
                        for path, var in model_state.items():
                            # Extract value from checkpoint_state using path tuple
                            val = ckpt_weights
                            try:
                                for key in path:
                                    # Handle string vs int keys (NNX List uses ints)
                                    if isinstance(val, dict):
                                        if key in val:
                                            val = val[key]
                                        elif str(key) in val:
                                            val = val[str(key)]
                                        else:
                                            raise KeyError(key)
                                    else:
                                        val = val[key]
                                
                                # If we found a leaf, update the variable
                                var.value = val
                            except (KeyError, TypeError, IndexError):
                                # Path not found, keep random init for this part
                                pass
                    else:
                        print(f"Warning: Restoration returned no weights, using random init.")
                else:
                    print(f"Warning: Restoration returned unexpected type {type(state)}, using random init.")
            except Exception as e:
                print(f"Restoration failed: {e}")
                print("Attempting minimal fallback with basic PyTreeCheckpointer...")
                checkpoint_dir = os.path.abspath(os.path.join(FLAGS.checkpoint, "default"))
                state = ocp.PyTreeCheckpointer().restore(checkpoint_dir)
                if state is not None and "ema" in state and state["ema"] is not None:
                    nnx.update(model, fix_keys(state["ema"]))
                elif state is not None and "model" in state:
                    nnx.update(model, fix_keys(state["model"]))
                
            print(f"Restored successfully.")
        else:
            print(f"Warning: Checkpoint {FLAGS.checkpoint} not found, using random init.")

        # 4. Prepare Attributes
        test_attributes = [a.strip() for a in FLAGS.attributes.split(",") if a.strip()]
        column_labels = ["Base"] + test_attributes
        num_cols = len(column_labels)
        num_rows = FLAGS.num_identities

        # 5. Sampling Setup
        sampler = DDIMSampler()
        vae_manager = VAEManager()
        
        # Ensure VAE is also on mesh
        vae_manager.params = jax.device_put(vae_manager.params, replicate_sharding)
        
        @nnx.jit
        def model_fn(x, t, y):
            out = model(x, t, y)
            if out.shape[-1] == x.shape[-1] * 2:
                return jnp.split(out, 2, axis=-1)[0]
            return out

        sample_shape = (
            num_rows,
            config.model.input_size,
            config.model.input_size,
            config.model.in_channels,
        )
        
        grid_images = [] # List of columns, each column is a batch of num_rows images

        print(f"Generating grid: {num_rows} identities x {num_cols} attribute states...")
        
        for attr_name in column_labels:
            print(f"  Sampling column: {attr_name}")
            
            # Construct label vector for this column
            y = jnp.zeros((num_rows, label_dim), dtype=jnp.int32)
            if attr_name != "Base":
                if attr_name not in CELEBA_ATTRIBUTE_NAMES:
                    raise ValueError(f"Unknown CelebA attribute: {attr_name}")
                attr_idx = CELEBA_ATTRIBUTE_NAMES.index(attr_name)
                y = y.at[:, attr_idx].set(1)
                
            null_y = jnp.zeros_like(y)
            
            # KEY: Use the SAME base_key for every column to ensure same initial noise
            samples = sampler.sample(
                model_fn, 
                sample_shape, 
                base_key, 
                num_inference_steps=FLAGS.num_steps,
                y=y,
                null_y=null_y,
                cfg_scale=FLAGS.cfg_scale
            )
            
            # Decode and store
            pixels = vae_manager.decode(samples)
            grid_images.append(np.asarray(pixels))

        # 6. Plot and Save
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
        
        # Handle single row/column edge cases
        if num_rows == 1 and num_cols == 1:
            axes_arr = np.array([[axes]])
        elif num_rows == 1:
            axes_arr = axes[np.newaxis, :]
        elif num_cols == 1:
            axes_arr = axes[:, np.newaxis]
        else:
            axes_arr = axes

        for c in range(num_cols):
            col_imgs = grid_images[c]
            axes_arr[0, c].set_title(column_labels[c], fontsize=16)
            for r in range(num_rows):
                axes_arr[r, c].imshow(col_imgs[r])
                axes_arr[r, c].axis("off")
                
        plt.tight_layout()
        plt.savefig(FLAGS.output_path, bbox_inches='tight')
        print(f"Saved attribute grid to {FLAGS.output_path}")
