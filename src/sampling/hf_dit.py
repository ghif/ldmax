"""Inference script to run pretrained Diffusion Transformers from Hugging Face with JAX/Flax."""

import os
import math
from typing import Any, Dict, Optional

from absl import flags
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from src.models.dit.dit import DiT
from src.training.sampler import DDIMSampler
from src.utils.vae import VAEManager
from src.utils.rng import RNGManager

FLAGS = flags.FLAGS
flags.DEFINE_string("model_id", "facebook/DiT-XL-2-256", "Hugging Face model ID.")
flags.DEFINE_integer("num_samples", 4, "Number of images to generate.")
flags.DEFINE_integer("num_steps", 50, "Number of sampling steps.")
flags.DEFINE_float("cfg_scale", 4.0, "Classifier-Free Guidance scale.")
flags.DEFINE_string("output_path", "./hf_samples.png", "Output file path.")
flags.DEFINE_integer("class_id", 207, "Class ID to sample (default 207 is Golden Retriever).")
flags.DEFINE_integer("seed", 42, "Random seed.")

def map_pt_to_flax(pt_state: Dict[str, np.ndarray], hidden_size: int, depth: int, learn_sigma: bool) -> Dict[str, Any]:
    """Map PyTorch Diffusers state dict to NNX DiT state dict."""
    flax_state = {}

    def transpose_linear(w):
        # PyTorch Linear is [out, in], JAX is [in, out]
        return w.T

    def map_layer(pt_key, flax_key, is_linear=True):
        if pt_key + ".weight" in pt_state:
            w = pt_state[pt_key + ".weight"]
            flax_state[flax_key + "/kernel"] = transpose_linear(w) if is_linear else w
        if pt_key + ".bias" in pt_state:
            flax_state[flax_key + "/bias"] = pt_state[pt_key + ".bias"]

    # 1. Patch Embedder (x_embedder)
    # PyTorch: pos_embed.proj.weight [hidden, 4, 2, 2]
    # JAX: x_embedder.kernel [16, hidden]
    if "pos_embed.proj.weight" in pt_state:
        w = pt_state["pos_embed.proj.weight"] # [hidden, c, p, p]
        # Transpose to [p, p, c, hidden] and reshape to [p*p*c, hidden]
        w_jax = w.transpose(2, 3, 1, 0).reshape(-1, hidden_size)
        flax_state["x_embedder/kernel"] = w_jax
    if "pos_embed.proj.bias" in pt_state:
        flax_state["x_embedder/bias"] = pt_state["pos_embed.proj.bias"]

    # 2. Timestep Embedder (t_embedder)
    map_layer("time_embedding.linear_1", "t_embedder/mlp/0")
    map_layer("time_embedding.linear_2", "t_embedder/mlp/2")

    # 3. Label Embedder (y_embedder)
    if "caption_projection.y_embedding" in pt_state:
        flax_state["y_embedder/embedding_table/embedding"] = pt_state["caption_projection.y_embedding"]
    elif "class_embedder.embedding_table.weight" in pt_state:
        flax_state["y_embedder/embedding_table/embedding"] = pt_state["class_embedder.embedding_table.weight"]

    # 4. Blocks
    for i in range(depth):
        pt_prefix = f"transformer_blocks.{i}"
        flax_prefix = f"blocks/{i}"

        # AdaLN modulation
        map_layer(f"{pt_prefix}.norm1.linear", f"{flax_prefix}/adaLN_modulation/layers/1")
        
        # LayerNorms
        map_layer(f"{pt_prefix}.norm1.norm", f"{flax_prefix}/norm1", is_linear=False)
        map_layer(f"{pt_prefix}.norm2.norm", f"{flax_prefix}/norm2", is_linear=False)

        # Attention
        # Diffusers often uses to_q, to_k, to_v
        if f"{pt_prefix}.attn1.to_q.weight" in pt_state:
            map_layer(f"{pt_prefix}.attn1.to_q", f"{flax_prefix}/attn/query")
            map_layer(f"{pt_prefix}.attn1.to_k", f"{flax_prefix}/attn/key")
            map_layer(f"{pt_prefix}.attn1.to_v", f"{flax_prefix}/attn/value")
            map_layer(f"{pt_prefix}.attn1.to_out.0", f"{flax_prefix}/attn/out")
        elif f"{pt_prefix}.attn1.to_qkv.weight" in pt_state:
            # Handle combined qkv
            qkv_w = pt_state[f"{pt_prefix}.attn1.to_qkv.weight"] # [3*hidden, hidden]
            qkv_b = pt_state[f"{pt_prefix}.attn1.to_qkv.bias"]
            q_w, k_w, v_w = np.split(qkv_w, 3, axis=0)
            q_b, k_b, v_b = np.split(qkv_b, 3, axis=0)
            
            flax_state[f"{flax_prefix}/attn/query/kernel"] = transpose_linear(q_w)
            flax_state[f"{flax_prefix}/attn/query/bias"] = q_b
            flax_state[f"{flax_prefix}/attn/key/kernel"] = transpose_linear(k_w)
            flax_state[f"{flax_prefix}/attn/key/bias"] = k_b
            flax_state[f"{flax_prefix}/attn/value/kernel"] = transpose_linear(v_w)
            flax_state[f"{flax_prefix}/attn/value/bias"] = v_b
            map_layer(f"{pt_prefix}.attn1.to_out.0", f"{flax_prefix}/attn/out")

        # MLP
        map_layer(f"{pt_prefix}.ff.net.0.proj", f"{flax_prefix}/mlp/layers/0")
        map_layer(f"{pt_prefix}.ff.net.2", f"{flax_prefix}/mlp/layers/2")

    # 5. Final Layer
    map_layer("norm_out.linear", "final_layer/adaLN_modulation/layers/1")
    map_layer("norm_out.norm", "final_layer/norm_final", is_linear=False)
    map_layer("proj_out", "final_layer/linear")

    return flax_state

def main(_):
    # 0. Construct Model and Load Weights
    print(f"Loading pretrained DiT from {FLAGS.model_id}...")
    
    # Construction parameters for DiT-XL/2 (the most common HF DiT)
    hidden_size = 1152
    depth = 28
    num_heads = 16
    patch_size = 2
    learn_sigma = True
    
    # Initialize NNX Model
    rng_manager = RNGManager(FLAGS.seed)
    model = DiT(
        input_size=32,
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        learn_sigma=learn_sigma,
        rngs=nnx.Rngs(rng_manager.next())
    )
    
    # Download and Load Weights
    try:
        weight_path = hf_hub_download(FLAGS.model_id, "diffusion_pytorch_model.safetensors", subfolder="transformer")
    except Exception:
        print("Safetensors not found, trying bin format...")
        weight_path = hf_hub_download(FLAGS.model_id, "diffusion_pytorch_model.bin", subfolder="transformer")
        # In a real scenario, we'd use torch.load here. Let's assume safetensors for now as it's common.
    
    pt_state = {}
    with safe_open(weight_path, framework="np", device="cpu") as f:
        for key in f.keys():
            pt_state[key] = f.get_tensor(key)
            
    print(f"Mapping weights for {len(pt_state)} tensors...")
    flax_params = map_pt_to_flax(pt_state, hidden_size, depth, learn_sigma)
    
    # Convert flat keys to nested dict for nnx.State
    nested_params = {}
    for k, v in flax_params.items():
        parts = k.split("/")
        curr = nested_params
        for p in parts[:-1]:
            if p.isdigit():
                p = int(p)
            if p not in curr:
                curr[p] = {}
            curr = curr[p]
        
        last_part = parts[-1]
        # In NNX, kernels and biases are inside Variable objects (e.g. Param)
        curr[last_part] = v

    # Update model state
    # We use nnx.State.from_flat to handle this more elegantly if we wanted, 
    # but manual nested update works too.
    # Actually, NNX state is basically a nested dict.
    state = nnx.state(model)
    
    def update_recursive(s_dict, p_dict):
        for k, v in p_dict.items():
            if isinstance(v, dict):
                update_recursive(s_dict[k], v)
            else:
                # Update the value of the NNX Variable
                s_dict[k].value = jnp.array(v)

    update_recursive(state, nested_params)
    nnx.update(model, state)
    print("Model weights loaded successfully.")

    # 1. Setup VAE and Sampler
    vae_manager = VAEManager() # Defaults to sd-vae-ft-mse-flax
    sampler = DDIMSampler()

    # 2. Run Inference
    print(f"Generating {FLAGS.num_samples} images for class {FLAGS.class_id}...")
    
    sample_labels = jnp.full((FLAGS.num_samples,), FLAGS.class_id, dtype=jnp.int32)
    null_labels = jnp.full((FLAGS.num_samples,), 1000, dtype=jnp.int32) # DiT-XL/2 has 1000 classes
    
    @nnx.jit
    def model_fn(x, t, y):
        out = model(x, t, y)
        if out.shape[-1] == x.shape[-1] * 2:
            return jnp.split(out, 2, axis=-1)[0]
        return out

    sample_shape = (FLAGS.num_samples, 32, 32, 4) # Latent shape for 256x256
    samples = sampler.sample(
        model_fn, 
        sample_shape, 
        rng_manager.next(), 
        num_inference_steps=FLAGS.num_steps,
        y=sample_labels,
        null_y=null_labels,
        cfg_scale=FLAGS.cfg_scale
    )
    
    print("Decoding latents...")
    images = vae_manager.decode(samples)
    
    # 3. Save Output
    grid_size = int(math.ceil(FLAGS.num_samples ** 0.5))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 4, grid_size * 4))
    if FLAGS.num_samples == 1:
        axes = [axes]
    for i, ax in enumerate(np.array(axes).flat):
        if i < FLAGS.num_samples:
            ax.imshow(np.asarray(images[i]))
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(FLAGS.output_path)
    print(f"Saved samples to {FLAGS.output_path}")
