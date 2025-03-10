import sys
import os
import glob
import argparse
import subprocess
import torch
import torch.nn as nn

sys.path.append("..")
from attention_lens.lens import Lens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default="/grand/SuperBERT/pettyjohnjn/AttnLens_GPT/checkpoint",
        type=str,
        help="Directory containing checkpoints for a lens",
    )
    parser.add_argument(
        "--save_dir",
        default="/grand/SuperBERT/pettyjohnjn/LoraLens/Full/ckpt_",
        type=str,
        help="Directory where extracted lenses will be saved",
    )
    return parser.parse_args()

def change_dict_key(d, old_key, new_key, default_value=None):
    """Rename key in dictionary."""
    d[new_key] = d.pop(old_key, default_value)

def merge_lora_weights(lens_instance):
    """
    Merge each LoRA layer's low-rank update with the shared base weight.
    For each layer, compute:
        effective_weight = shared_unembed + (lora_B @ lora_A) * (lora_alpha / r)
    and replace the LoRA layer with a standard nn.Linear.
    """
    merged_linears = []
    for idx, layer in enumerate(lens_instance.linears):
        if not (hasattr(layer, 'lora_A') and hasattr(layer, 'lora_B')):
            raise AttributeError(f"Layer {idx} is missing LoRA parameters.")

        lora_A = layer.lora_A  # shape: (r, d_model)
        lora_B = layer.lora_B  # shape: (d_vocab, r)
        r_val = lora_A.shape[0]
        lora_alpha = getattr(layer, 'lora_alpha', 1)
        scaling = lora_alpha / r_val if r_val != 0 else 1.0

        update = torch.matmul(lora_B, lora_A) * scaling
        base = lens_instance.shared_unembed.detach()
        effective_weight = base + update.detach()

        merged_layer = nn.Linear(lens_instance.d_model, lens_instance.d_vocab)
        merged_layer.weight = nn.Parameter(effective_weight)
        merged_layer.bias = nn.Parameter(layer.bias.detach().clone())
        merged_linears.append(merged_layer)

    lens_instance.linears = nn.ModuleList(merged_linears)
    lens_instance.shared_unembed = None  # no longer needed
    return lens_instance

def infer_config_from_state(state_dict):
    """
    Infer d_model, d_vocab, and n_layers from the checkpoint state.
      - shared_unembed is expected to have shape (d_vocab, d_model)
      - The number of layers is inferred by the highest index in keys like "linears.X.lora_A"
    """
    if "shared_unembed" not in state_dict:
        raise KeyError("Checkpoint does not contain 'shared_unembed'.")
    shared_unembed = state_dict["shared_unembed"]
    d_vocab, d_model = shared_unembed.shape

    # Infer n_layers from keys such as "linears.0.lora_A"
    layer_indices = set()
    for key in state_dict.keys():
        if key.startswith("linears."):
            try:
                # key format: "linears.{idx}.<param>"
                parts = key.split(".")
                layer_indices.add(int(parts[1]))
            except (IndexError, ValueError):
                continue
    n_layers = max(layer_indices) + 1 if layer_indices else 0
    return d_model, d_vocab, n_layers

def load_checkpoint_state(ckpt_path):
    """
    Load and clean the checkpoint state.
      - If ckpt_path is a directory, runs conversion to fp32.
      - Removes unwanted prefixes and retains keys starting with "attn_lens".
    """
    if os.path.isdir(ckpt_path):
        print(f"{ckpt_path} is a directory. Converting using zero_to_fp32...")
        conversion_script = os.path.join(ckpt_path, "zero_to_fp32.py")
        if not os.path.exists(conversion_script):
            raise FileNotFoundError(f"{conversion_script} not found.")
        fp32_ckpt = os.path.join(ckpt_path, "pytorch_model_fp32.bin")
        subprocess.run(
            ["python", conversion_script, ".", "pytorch_model_fp32.bin"],
            check=True,
            cwd=ckpt_path,
        )
        ckpt_to_load = fp32_ckpt
    else:
        ckpt_to_load = ckpt_path

    print(f"Loading checkpoint from {ckpt_to_load}")
    ckpt = torch.load(ckpt_to_load, map_location="cpu",weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)

    # Clean up keys: remove any prefix (e.g., "_forward_module.") and the "attn_lens" prefix.
    for key in list(state_dict.keys()):
        clean_key = key.replace("_forward_module.", "")
        if not clean_key.startswith("attn_lens"):
            del state_dict[key]
        else:
            # Remove the "attn_lens" prefix (assumed to be 10 characters long).
            new_key = clean_key[10:]
            change_dict_key(state_dict, key, new_key)
    return state_dict

def build_lens_from_checkpoint(ckpt_path, r=8, lora_alpha=1, lora_dropout=0.0, merge_weights=True):
    """
    Build a new lens instance directly from the checkpoint.
      - Infers model dimensions from the saved state.
      - Uses the transposed shared_unembed as the dummy "unembed".
      - Uses the bias from the first layer (if available) or zeros.
    """
    state_dict = load_checkpoint_state(ckpt_path)
    d_model, d_vocab, n_layers = infer_config_from_state(state_dict)
    print(f"Inferred config -- d_model: {d_model}, d_vocab: {d_vocab}, n_layers: {n_layers}")

    # Use the transposed shared_unembed as unembed (since shared_unembed is unembed.T).
    shared_unembed = state_dict["shared_unembed"]
    unembed = shared_unembed.t().clone()  # shape: (d_model, d_vocab)

    # Use bias from the first layer if available; otherwise create zeros.
    first_bias_key = "linears.0.bias"
    if first_bias_key in state_dict:
        bias = state_dict[first_bias_key].clone()
    else:
        bias = torch.zeros(d_vocab)

    # Instantiate the new lens. The lens class (LensLR) is obtained by name.
    lens_cls_name = "lenslr"
    lens_cls = Lens.get_lens(lens_cls_name)
    lens_instance = lens_cls(
        unembed=nn.Parameter(unembed),
        bias=nn.Parameter(bias),
        n_layers=n_layers,
        d_model=d_model,
        d_vocab=d_vocab,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        merge_weights=merge_weights,
    )

    # Update each layer's parameters (lora_A, lora_B, bias) from the checkpoint.
    for layer_idx in range(n_layers):
        prefix = f"linears.{layer_idx}."
        for param_name in ["lora_A", "lora_B", "bias"]:
            key = prefix + param_name
            if key in state_dict:
                param = state_dict[key]
                if param_name == "bias":
                    lens_instance.linears[layer_idx].bias = nn.Parameter(param)
                elif param_name == "lora_A":
                    lens_instance.linears[layer_idx].lora_A = nn.Parameter(param)
                elif param_name == "lora_B":
                    lens_instance.linears[layer_idx].lora_B = nn.Parameter(param)
            else:
                print(f"Warning: '{key}' not found in checkpoint.")
    return lens_instance

def extract_and_save_lens(ckpt_path, save_path, r=8, lora_alpha=1, lora_dropout=0.0, merge_weights=True):
    """
    Build the lens from the checkpoint, merge its LoRA weights, and save it.
    """
    lens_instance = build_lens_from_checkpoint(
        ckpt_path, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights
    )
    merge_lora_weights(lens_instance)
    torch.save(lens_instance, save_path)
    print(f"Extracted lens saved to {save_path}")

def process_checkpoints(ckpt_dir, save_dir, r=8, lora_alpha=1, lora_dropout=0.0, merge_weights=True):
    """
    Recursively process checkpoint files in ckpt_dir,
    extract lens for each, and save them in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "**/*.ckpt"), recursive=True)
    for ckpt_file in ckpt_files:
        print(f"Processing checkpoint: {ckpt_file}")
        save_path = os.path.join(save_dir, os.path.basename(ckpt_file))
        extract_and_save_lens(
            ckpt_file, save_path, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights
        )

def main():
    args = parse_args()
    print("Current working directory:", os.getcwd())
    process_checkpoints(args.ckpt_dir, args.save_dir)
    print("Extraction process complete.")

if __name__ == "__main__":
    main()