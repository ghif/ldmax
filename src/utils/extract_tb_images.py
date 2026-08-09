"""Script to extract images from TensorBoard logs."""

import os
import io
import numpy as np
from absl import flags
from tensorboard.backend.event_processing import event_accumulator
from PIL import Image
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("tb_log_dir", "", "Path to the TensorBoard log directory.")
flags.DEFINE_string("output_dir", "./extracted_images", "Directory to save the extracted images.")
flags.DEFINE_string("tag", "train/samples", "Tag of the images to extract (or 'all').")
flags.DEFINE_boolean("make_gif", False, "Whether to also create a GIF from the extracted images per tag.")
flags.DEFINE_integer("gif_fps", 5, "FPS for the generated GIF.")

def main(_):
    if not FLAGS.tb_log_dir:
        raise ValueError("Please provide a --tb_log_dir")
        
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    
    print(f"Loading TensorBoard events from: {FLAGS.tb_log_dir}")
    # size_guidance=0 means load all events
    ea = event_accumulator.EventAccumulator(
        FLAGS.tb_log_dir,
        size_guidance={event_accumulator.IMAGES: 0}
    )
    ea.Reload()
    
    image_tags = ea.Tags().get('images', [])
    if not image_tags:
        print("No image tags found in the log directory.")
        return
        
    print(f"Found image tags: {image_tags}")
    
    if FLAGS.tag == "all":
        tags_to_extract = image_tags
    else:
        # Match exact tag or common prefixes (e.g., if TB appends /0, /1)
        tags_to_extract = [t for t in image_tags if t == FLAGS.tag or t.startswith(FLAGS.tag + "/")]
        
        if not tags_to_extract:
            print(f"No image tags matching '{FLAGS.tag}' found.")
            return

    for tag in tags_to_extract:
        print(f"Extracting images for tag: {tag}")
        events = ea.Images(tag)
        
        tag_safe = tag.replace("/", "_")
        tag_dir = os.path.join(FLAGS.output_dir, tag_safe)
        os.makedirs(tag_dir, exist_ok=True)
        
        frames = []
        for event in tqdm(events, desc="Extracting"):
            step = event.step
            # event.encoded_image_string contains the raw bytes
            try:
                img = Image.open(io.BytesIO(event.encoded_image_string))
                out_path = os.path.join(tag_dir, f"step_{step:08d}.png")
                img.save(out_path)
                
                if FLAGS.make_gif:
                    # Keep track of frames for the GIF
                    frames.append(img.copy())
            except Exception as e:
                print(f"Failed to extract image at step {step} for tag {tag}: {e}")
                
        if FLAGS.make_gif and frames:
            gif_path = os.path.join(FLAGS.output_dir, f"{tag_safe}.gif")
            print(f"Saving GIF to {gif_path}")
            # Use PIL to save animated GIF
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=1000 // FLAGS.gif_fps,
                loop=0
            )
