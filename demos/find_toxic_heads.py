#!/usr/bin/env python
import sys
import os
import re
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

sys.path.append("..")


def load_toxic_words(filepath):
    """
    Load toxic words from a file.
    
    Returns:
        A set of toxic words (all in lower case).
    """
    with open(filepath, 'r') as file:
        return set(word.strip().lower() for word in file.read().splitlines())


def create_and_save_heatmap(data, layer_nums, pdf_filename, title):
    """
    Create and save a heatmap from the given data.
    
    Args:
        data (np.ndarray): 2D array to plot.
        layer_nums (list): List of layer indices.
        pdf_filename (str): Output PDF file path.
        title (str): Title of the heatmap.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_xticklabels(range(data.shape[1]))
    ax.set_yticklabels([f'Layer {layer}' for layer in layer_nums])
    ax.set_xlabel('Attention Head')
    ax.set_ylabel('Layer')
    ax.set_title(title)
    plt.tight_layout()

    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def interpret_prompt(prompt, attn_lens, common_toxic_tokens, total_toxic_counts, total_confidence,
                     tokenizer, toxic_words, device, num_attn_heads, k_tokens, model):
    """
    Process a single prompt to obtain token probabilities and toxic token counts.
    
    For each layer, every head's output is passed through the corresponding layer's lens.
    The resulting probabilities are used to accumulate confidence scores and count toxic tokens.
    
    Args:
        prompt (str): Input text prompt.
        attn_lens: Loaded attention lens checkpoint.
        common_toxic_tokens: List of Counters per (layer, head) for toxic token counts.
        total_toxic_counts: 2D np.ndarray storing toxic counts per (layer, head).
        total_confidence: 2D np.ndarray storing confidence sums per (layer, head).
        tokenizer: Hugging Face tokenizer.
        toxic_words (set): Set of toxic words.
        device (str): Device identifier ("cpu" or "cuda").
        num_attn_heads (int): Number of attention heads.
        k_tokens (int): Top k token predictions to consider.
        model: Loaded language model.
    """
    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run the model to ensure transformer caches are populated.
    with torch.no_grad():
        _ = model(**inputs)

        # Iterate over each layer.
        for layer in range(attn_lens.n_layers):
            # Retrieve cached outputs for the current layer.
            # Expected shape: [batch, seq, num_heads, d_model_head]
            cache = model.transformer.h[layer].attn.head_out  
            # Get the single lens for this layer.
            lens_layer = attn_lens.linears[layer]
            
            # For each head, project its output and process predictions.
            for head in range(num_attn_heads):
                head_output = cache[0, -1, head]  # Use last token for the first batch element.
                projected = lens_layer(head_output)
                probs = torch.softmax(projected, dim=-1)
                topk_probs, topk_token_preds = torch.topk(probs, k_tokens)
                confidence_sum = topk_probs.sum().item()
                total_confidence[layer, head] += confidence_sum

                topk_token_ids = topk_token_preds.cpu().numpy().tolist()
                toxic_tokens_found = []
                for token_id in topk_token_ids:
                    token_str = tokenizer.decode(token_id).strip().lower()
                    if token_str in toxic_words:
                        toxic_tokens_found.append((token_str, token_id))
                toxic_count = len(toxic_tokens_found)
                total_toxic_counts[layer, head] += toxic_count
                common_toxic_tokens[layer][head].update(toxic_tokens_found)


def load_all_layers_lens(lens_folder, device):
    """
    Load the single checkpoint file containing the attention lens for all layers.
    
    Assumes there is one .ckpt file in the specified folder.
    
    Returns:
        The loaded attention lens checkpoint.
    """
    ckpt_files = [f for f in os.listdir(lens_folder) if f.endswith('.ckpt')]
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint file found in the lens folder.")
    ckpt_path = os.path.join(lens_folder, ckpt_files[0])
    print(f"Loading all-layers lens checkpoint from {ckpt_path}")
    return torch.load(ckpt_path, map_location=torch.device(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", type=str)
    parser.add_argument("--lense_folder", default="/path/to/lens_folder", type=str,
                        help="Folder containing the checkpoint for all layers")
    parser.add_argument("--layer_num", nargs='+', default=list(range(12)), type=int)
    parser.add_argument("--num_attn_heads", default=12, choices=[12, 20], type=int)
    parser.add_argument("--k_tokens", default=50, type=int)
    parser.add_argument("--cpu", default=True, type=bool)
    parser.add_argument("--toxic_dict_path", default="toxic_dictionary.txt", type=str)
    parser.add_argument("--output_folder", default="outputs", type=str,
                        help="Folder to place all output files")
    parser.add_argument("--output_toxic_pdf", default="heatmaps_toxic.pdf", type=str,
                        help="Output PDF file for toxic tokens heatmap")
    parser.add_argument("--output_confidence_pdf", default="heatmaps_confidence.pdf", type=str,
                        help="Output PDF file for confidence heatmap")
    parser.add_argument("--output_txt", default="best_toxic_tokens.txt", type=str,
                        help="Output TXT file for toxic token dictionary")
    parser.add_argument("--output_toxic_npy", default="best_average_toxic_counts.npy", type=str,
                        help="Output Numpy file for average toxic counts")
    parser.add_argument("--output_confidence_npy", default="lens_confidence.npy", type=str,
                        help="Output Numpy file for lens confidence heatmap")
    args = parser.parse_args()

    # Ensure output folder exists.
    os.makedirs(args.output_folder, exist_ok=True)
    output_toxic_pdf_path = os.path.join(args.output_folder, args.output_toxic_pdf)
    output_confidence_pdf_path = os.path.join(args.output_folder, args.output_confidence_pdf)
    output_txt_path = os.path.join(args.output_folder, args.output_txt)
    output_toxic_npy_path = os.path.join(args.output_folder, args.output_toxic_npy)
    output_confidence_npy_path = os.path.join(args.output_folder, args.output_confidence_npy)

    # Set up device.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.cpu:
        device = "cpu"

    # Load tokenizer, config, and model.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config).to(device)
    
    # Load toxic words.
    toxic_words = load_toxic_words(args.toxic_dict_path)

    # Load dataset and filter toxic prompts.
    dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
    toxic_prompts = dataset.filter(lambda example: example['label'] == 1)['comment_text'][:16]

    # Load the single all-layers attention lens checkpoint.
    attn_lens = load_all_layers_lens(args.lense_folder, device)
    n_layers = attn_lens.n_layers
    num_heads = args.num_attn_heads

    # Initialize result arrays.
    total_toxic_counts = np.zeros((n_layers, num_heads))
    total_confidence = np.zeros((n_layers, num_heads))
    common_toxic_tokens = [[Counter() for _ in range(num_heads)] for _ in range(n_layers)]

    # Process each toxic prompt.
    for prompt in tqdm(toxic_prompts, desc="Processing Toxic Prompts"):
        interpret_prompt(prompt, attn_lens, common_toxic_tokens, total_toxic_counts, total_confidence,
                         tokenizer, toxic_words, device, num_heads, args.k_tokens, model)

    # Calculate averages.
    average_toxic_counts = total_toxic_counts / len(toxic_prompts)
    average_confidence = total_confidence / len(toxic_prompts)

    # Write toxic tokens to file.
    with open(output_txt_path, 'w') as f:
        for layer in range(n_layers):
            f.write(f"\nLayer {layer}:\n")
            for head in range(num_heads):
                most_common_tokens = common_toxic_tokens[layer][head].most_common(10)
                f.write(f"  Head {head}:\n")
                for (token_str, token_id), count in most_common_tokens:
                    f.write(f"    Token: {token_str}, Count: {count}, ID: {token_id}\n")

    # Create and save heatmaps.
    create_and_save_heatmap(average_toxic_counts, list(range(n_layers)), output_toxic_pdf_path,
                            'Average Toxic Tokens per Head per Layer')
    create_and_save_heatmap(average_confidence, list(range(n_layers)), output_confidence_pdf_path,
                            'Average Confidence per Head per Layer')

    # Save numpy arrays.
    np.save(output_toxic_npy_path, average_toxic_counts)
    np.save(output_confidence_npy_path, average_confidence)

    print("Processing complete.")


if __name__ == "__main__":
    main()