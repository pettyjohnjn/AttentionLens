import sys
sys.path.append("..")

from attention_lens.model.get_model import get_model
from attention_lens.lens import Lens
import torch
import torch.nn as nn
import glob
import os
import argparse
import subprocess

# Print the current working directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Set up user args
parser = argparse.ArgumentParser()

parser.add_argument(
    "--ckpt_dir",
    default="/grand/SuperBERT/pettyjohnjn/AttnLens_GPT/checkpoint_rank02",
    type=str,
    help="Path to directory containing all latest ckpts for a lens",
)

parser.add_argument(
    "--save_dir",
    default="/grand/SuperBERT/pettyjohnjn/LoraLens_Pile2/extracted_checkpoint/gpt2/R02/",
    type=str,
    help="Path to directory where script should save all extracted lenses",
)

args = parser.parse_args()

# Single device
device = "cpu"

# Initialize the model (and get its unembed weight for the lens)
model, _ = get_model(device=device)
bias = torch.zeros(50257).to(device)

# For the "lenslr" version, retrieve the appropriate class using its name.
lens_cls_name = "lenslr"
lens_cls = Lens.get_lens(lens_cls_name)

# Instantiate the attention lens using the LensLR constructor.
attn_lens = lens_cls(
    unembed=model.lm_head.weight.T,
    bias=bias,
    n_head=model.config.num_attention_heads,
    d_model=model.config.hidden_size,
    d_vocab=model.config.vocab_size,
    r=8,                 # LoRA rank
    lora_alpha=1,        # LoRA alpha scaling
    lora_dropout=0.0,    # LoRA dropout probability
    merge_weights=True,  # (We will merge explicitly below.)
)

def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)

def merge_lora_weights(attn_lens):
    """
    For each head, compute:
        effective_weight = shared_unembed + (lora_B @ lora_A) * (lora_alpha / r)
    and then replace the LoRA layer with a standard nn.Linear holding the merged weight and bias.
    """
    new_linears = []
    for i, linear in enumerate(attn_lens.linears):
        if not (hasattr(linear, 'lora_A') and hasattr(linear, 'lora_B')):
            raise AttributeError(f"Linear layer {i} is missing LoRA parameters.")
        
        lora_A = linear.lora_A  # expected shape: (r, d_model)
        lora_B = linear.lora_B  # expected shape: (d_vocab, r)
        r_val = lora_A.shape[0]
        lora_alpha = getattr(linear, 'lora_alpha', 1)
        scaling = lora_alpha / r_val if r_val != 0 else 1.0

        update = torch.matmul(lora_B, lora_A) * scaling
        base = attn_lens.shared_unembed.detach()
        effective_weight = base + update.detach()

        bias_linear = linear.bias.detach().clone()
        new_linear = nn.Linear(attn_lens.d_model, attn_lens.d_vocab)
        new_linear.weight = nn.Parameter(effective_weight)
        new_linear.bias = nn.Parameter(bias_linear)
        new_linears.append(new_linear)

    attn_lens.linears = nn.ModuleList(new_linears)
    attn_lens.shared_unembed = None
    return attn_lens

def extract_and_save_lense_from_ckpt(ckpt_filepath, save_filepath):
    if os.path.isdir(ckpt_filepath):
        print(f"{ckpt_filepath} is a directory. Running zero_to_fp32 conversion.")
        zero_to_fp32_script = os.path.join(ckpt_filepath, "zero_to_fp32.py")
        if not os.path.exists(zero_to_fp32_script):
            print(f"Error: {zero_to_fp32_script} not found in the checkpoint directory.")
            return

        fp32_ckpt = os.path.join(ckpt_filepath, "pytorch_model_fp32.bin")
        command = ["python", zero_to_fp32_script, ".", "pytorch_model_fp32.bin"]
        print("Running command:", " ".join(command))
        subprocess.run(command, check=True, cwd=ckpt_filepath)
        ckpt_to_load = fp32_ckpt
    else:
        ckpt_to_load = ckpt_filepath

    print(f"Loading checkpoint from {ckpt_to_load}")
    loaded_ckpt = torch.load(ckpt_to_load, map_location="cpu")
    a = loaded_ckpt["state_dict"] if "state_dict" in loaded_ckpt else loaded_ckpt

    for key in list(a.keys()):
        clean_key = key.replace("_forward_module.", "")
        if not clean_key.startswith("attn_lens"):
            del a[key]
        else:
            new_key = clean_key[10:]
            change_dict_key(a, key, new_key)

    if "shared_unembed" in a:
        attn_lens.shared_unembed = nn.Parameter(a["shared_unembed"])
    else:
        print("Warning: 'shared_unembed' not found in checkpoint state_dict.")

    for i in range(attn_lens.n_head):
        prefix = f"linears.{i}."
        for sub in ["lora_A", "lora_B", "bias"]:
            key = prefix + sub
            if key in a:
                param = a[key]
                if sub == "bias":
                    attn_lens.linears[i].bias = nn.Parameter(param)
                elif sub == "lora_A":
                    attn_lens.linears[i].lora_A = nn.Parameter(param)
                elif sub == "lora_B":
                    attn_lens.linears[i].lora_B = nn.Parameter(param)
            else:
                print(f"Warning: '{key}' not found in checkpoint state_dict.")

    merge_lora_weights(attn_lens)

    print(f"Saving extracted lens to {save_filepath}")
    torch.save(attn_lens, save_filepath)
    print(f"Successfully saved extracted lens to {save_filepath}")

def iter_thru_ckpts_extract_lenses(ckpt_dir, save_dir):
    # Recursively iterate through all .ckpt files (or directories) in ckpt_dir.
    for filename in glob.glob(os.path.join(ckpt_dir, "**/*.ckpt"), recursive=True):
        print(f"Processing checkpoint: {filename}")

        # Determine the layer number from the parent folder's name.
        parent_dir = os.path.basename(os.path.dirname(filename))
        # Expecting folder names like "ckpt_02". Adjust if your naming differs.
        if parent_dir.startswith("ckpt_"):
            layer_num = parent_dir.split("ckpt_")[-1]
        else:
            layer_num = "unknown"
        
        # Create a subfolder in save_dir for this layer.
        layer_folder = os.path.join(save_dir, f"L{layer_num}")
        if not os.path.exists(layer_folder):
            os.makedirs(layer_folder)
        
        # Save file in the appropriate layer folder.
        save_filepath = os.path.join(layer_folder, os.path.basename(filename))
        extract_and_save_lense_from_ckpt(filename, save_filepath=save_filepath)

print("Starting extraction process...")
iter_thru_ckpts_extract_lenses(args.ckpt_dir, args.save_dir)
print("Done")