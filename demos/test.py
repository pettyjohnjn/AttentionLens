import sys
import os
import argparse
import glob
import torch

# Add path to custom modules
sys.path.append("..")

from attention_lens.model.get_model import get_model
from attention_lens.lens import Lens

# Print the current working directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Set up user arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--ckpt_dir",
    default="/grand/SuperBERT/pettyjohnjn/checkpoint/test",
    type=str,
    help="Path to dir containing all latest ckpts for a lens",
)
parser.add_argument(
    "--save_dir",
    default="/grand/SuperBERT/pettyjohnjn/extracted_checkpoint/gpt2/ckpt_8/",
    type=str,
    help="Path to dir where script should save all extracted lenses",
)
args = parser.parse_args()

# Single device
device = "cpu"

# Initialize lens model
model, _ = get_model(device=device)

# Initialize attention lens with necessary parameters
bias = torch.zeros(50257).to(device)  # Using a zeroed bias for initialization
lens_cls = Lens.get_lens("lensa")
attn_lens = lens_cls(
    unembed=model.lm_head.weight.T,
    bias=bias,
    n_head=model.config.num_attention_heads,
    d_model=model.config.hidden_size,
    d_vocab=model.config.vocab_size,
)

# Function to rename keys if necessary
def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)

# Function to extract and save lens from checkpoint
def extract_and_save_lens_from_ckpt(ckpt_filepath, save_filepath):
    print(f"Loading checkpoint from {ckpt_filepath}")
    
    # Load checkpoint (DeepSpeed format with weights only)
    checkpoint = torch.load(ckpt_filepath, map_location="cpu")

    # Filter for attention lens parameters only
    attn_lens_params = {k[10:]: v for k, v in checkpoint.items() if k.startswith("attn_lens")}
    
    # Load state dict into attention lens
    print(f"Loading state dict into attention lens from {ckpt_filepath}")
    attn_lens.load_state_dict(attn_lens_params, strict=False)
    
    # Save the extracted attention lens
    print(f"Saving extracted lens to {save_filepath}")
    torch.save(attn_lens, save_filepath)
    print(f"Successfully saved extracted lens to {save_filepath}")

# Function to iterate through checkpoints and extract lenses
def iter_thru_ckpts_extract_lenses(ckpt_dir, save_dir):
    for filename in glob.glob(os.path.join(ckpt_dir, "**/*.ckpt"), recursive=True):
        print(f"Processing checkpoint: {filename}")
        save_filepath = os.path.join(save_dir, os.path.basename(filename))

        # If save_dir doesn't exist, create it
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        extract_and_save_lens_from_ckpt(filename, save_filepath=save_filepath)

print("Starting extraction process...")
iter_thru_ckpts_extract_lenses(args.ckpt_dir, args.save_dir)
print("Done")