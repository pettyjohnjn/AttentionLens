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
    default="/grand/SuperBERT/pettyjohnjn/AttentionLens/checkpoint",
    type=str,
    help="Path to directory containing all latest ckpts for a lens",
)

parser.add_argument(
    "--save_dir",
    default="/grand/SuperBERT/pettyjohnjn/LoraLens_Pile2/extracted_checkpoint/gpt2/ckpt_8",
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
        # Here we expect each linear to have lora_A and lora_B.
        # Instead of relying on linear.r, compute the LoRA rank from lora_A.
        if not (hasattr(linear, 'lora_A') and hasattr(linear, 'lora_B')):
            raise AttributeError(f"Linear layer {i} is missing LoRA parameters.")
        
        lora_A = linear.lora_A  # expected shape: (r, d_model)
        lora_B = linear.lora_B  # expected shape: (d_vocab, r)
        
        # Compute r as the number of rows in lora_A.
        r_val = lora_A.shape[0]
        
        # Try to retrieve lora_alpha; if missing, default to 1.
        lora_alpha = getattr(linear, 'lora_alpha', 1)
        scaling = lora_alpha / r_val if r_val != 0 else 1.0

        # Compute the low-rank update: (lora_B @ lora_A) * scaling.
        update = torch.matmul(lora_B, lora_A) * scaling

        # The shared base weight is stored in attn_lens.shared_unembed.
        base = attn_lens.shared_unembed.detach()
        effective_weight = base + update.detach()

        # Get the bias from the current linear layer.
        bias_linear = linear.bias.detach().clone()

        # Create a new standard nn.Linear layer with merged weight and bias.
        new_linear = nn.Linear(attn_lens.d_model, attn_lens.d_vocab)
        new_linear.weight = nn.Parameter(effective_weight)
        new_linear.bias = nn.Parameter(bias_linear)

        new_linears.append(new_linear)

    # Replace the LoRA linear layers with the new merged layers.
    attn_lens.linears = nn.ModuleList(new_linears)
    # Optionally, remove the now-unneeded shared_unembed.
    attn_lens.shared_unembed = None
    return attn_lens

def extract_and_save_lense_from_ckpt(ckpt_filepath, save_filepath):
    # If the provided ckpt is a directory (DeepSpeed checkpoint), run the conversion script.
    if os.path.isdir(ckpt_filepath):
        print(f"{ckpt_filepath} is a directory. Running zero_to_fp32 conversion.")
        zero_to_fp32_script = os.path.join(ckpt_filepath, "zero_to_fp32.py")
        if not os.path.exists(zero_to_fp32_script):
            print(f"Error: {zero_to_fp32_script} not found in the checkpoint directory.")
            return

        # Use a temporary filename for the consolidated fp32 checkpoint.
        fp32_ckpt = os.path.join(ckpt_filepath, "pytorch_model_fp32.bin")
        command = ["python", zero_to_fp32_script, ".", "pytorch_model_fp32.bin"]
        print("Running command:", " ".join(command))
        subprocess.run(command, check=True, cwd=ckpt_filepath)
        ckpt_to_load = fp32_ckpt
    else:
        ckpt_to_load = ckpt_filepath

    print(f"Loading checkpoint from {ckpt_to_load}")
    loaded_ckpt = torch.load(ckpt_to_load, map_location="cpu")
    # If the checkpoint is wrapped in a dict under "state_dict", extract it.
    a = loaded_ckpt["state_dict"] if "state_dict" in loaded_ckpt else loaded_ckpt

    # Filter out and re-map keys: remove any unwanted prefixes.
    # For example, if keys are like "_forward_module.attn_lens.shared_unembed", remove the prefix.
    for key in list(a.keys()):
        clean_key = key.replace("_forward_module.", "")
        if not clean_key.startswith("attn_lens"):
            del a[key]
        else:
            # Remove the "attn_lens" prefix (assumed to be 10 characters long).
            new_key = clean_key[10:]
            change_dict_key(a, key, new_key)

    # --- Instead of using attn_lens.load_state_dict(a) (which raises unexpected key errors),
    # we manually update the model parameters using the checkpoint values.
    #
    # The checkpoint (after filtering) is expected to contain keys:
    #   "shared_unembed"
    #   "linears.0.lora_A", "linears.0.lora_B", "linears.0.bias", etc.
    #
    # We update the corresponding attributes of attn_lens.
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

    # Merge the LoRA low-rank parameters with the shared base weight.
    merge_lora_weights(attn_lens)

    print(f"Saving extracted lens to {save_filepath}")
    torch.save(attn_lens, save_filepath)
    print(f"Successfully saved extracted lens to {save_filepath}")

def iter_thru_ckpts_extract_lenses(ckpt_dir, save_dir):
    # Recursively iterate through all .ckpt files (or directories) in ckpt_dir.
    for filename in glob.glob(os.path.join(ckpt_dir, "**/*.ckpt"), recursive=True):
        print(f"Processing checkpoint: {filename}")
        save_filepath = os.path.join(save_dir, os.path.basename(filename))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        extract_and_save_lense_from_ckpt(filename, save_filepath=save_filepath)

print("Starting extraction process...")
iter_thru_ckpts_extract_lenses(args.ckpt_dir, args.save_dir)
print("Done")