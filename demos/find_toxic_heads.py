import sys
sys.path.append("..")
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

# Load toxic words from a file
def load_toxic_words(filepath):
    with open(filepath, 'r') as file:
        return set(word.strip().lower() for word in file.read().splitlines())

# Create and save heatmap with a custom title
def create_and_save_heatmap(data, layer_nums, pdf_filename, title):
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_xticklabels(range(data.shape[1]))
    ax.set_yticklabels([f'Layer {layer_num}' for layer_num in layer_nums])
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.title(title)

    plt.tight_layout()
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

# Process each prompt through the model and extract toxic tokens and confidence from attention heads.
# This version records toxic tokens as a tuple (token, token_id) if they are in the toxic words list.
def interpret_prompt(prompt, attn_lenses, common_toxic_tokens, total_toxic_counts, total_confidence, tokenizer, toxic_words, device, num_attn_heads, k_tokens):
    inputs = tokenizer(prompt, truncation=True, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        for layer_num, attn_lens in enumerate(attn_lenses):
            cache = model.transformer.h[layer_num].attn.head_out
            inputs_list = [cache]
            input_tensor = torch.stack(inputs_list)

            for head in range(num_attn_heads):
                layer_head = attn_lens.linears[head]
                projected = layer_head(input_tensor[0][0][-1][head])
                # Compute softmax to get probabilities
                probs = torch.softmax(projected, dim=-1)
                # Get top k tokens (both probabilities and token indices)
                topk_probs, topk_token_preds = torch.topk(probs, k_tokens)
                # Sum the probabilities for the top tokens
                confidence_sum = topk_probs.sum().item()
                total_confidence[layer_num][head] += confidence_sum

                # Get token ids directly from topk results
                topk_token_ids = topk_token_preds.cpu().numpy().tolist()

                toxic_tokens_found = []
                for token_id in topk_token_ids:
                    token_str = tokenizer.decode(token_id).strip().lower()
                    if token_str in toxic_words:
                        # Record as a tuple: (token string, token id)
                        toxic_tokens_found.append((token_str, token_id))
                
                toxic_count = len(toxic_tokens_found)
                total_toxic_counts[layer_num][head] += toxic_count
                common_toxic_tokens[layer_num][head].update(toxic_tokens_found)

# Function to load the best checkpoint from each layer folder
def load_best_lens_checkpoints(lens_folder):
    best_checkpoints = []
    # Iterate over each subfolder (e.g., L0, L1, ...)
    for subfolder in sorted(os.listdir(lens_folder)):
        subfolder_path = os.path.join(lens_folder, subfolder)
        if os.path.isdir(subfolder_path):
            ckpt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.ckpt')]
            best_loss = float('inf')
            best_ckpt = None
            for ckpt in ckpt_files:
                # Extract train_loss value from the filename
                match = re.search(r"train_loss=([\d\.]+)", ckpt)
                if match:
                    loss = float(match.group(1).rstrip('.'))
                    if loss < best_loss:
                        best_loss = loss
                        best_ckpt = ckpt
            if best_ckpt is not None:
                best_checkpoints.append(os.path.join(subfolder_path, best_ckpt))
    return best_checkpoints

# Set up user arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="gpt2", type=str)
# Instead of individual checkpoint paths, now provide a folder containing subfolders of checkpoints.
parser.add_argument("--lense_folder", default="/path/to/lens_folder", type=str, help="Folder containing subfolders with .ckpt files for each layer")
parser.add_argument("--lens", default="gpt2", choices=["gpt2"], type=str)
parser.add_argument("--layer_num", nargs='+', default=list(range(12)), type=int)
parser.add_argument("--num_attn_heads", default=12, choices=[12, 20], type=int)
parser.add_argument("--k_tokens", default=50, type=int)
parser.add_argument("--cpu", default=True, type=bool)
parser.add_argument("--toxic_dict_path", default="toxic_dictionary.txt", type=str)
parser.add_argument("--output_folder", default="outputs", type=str, help="Folder to place all output files")
parser.add_argument("--output_toxic_pdf", default="heatmaps_toxic.pdf", type=str, help="Output PDF file for toxic tokens heatmap")
parser.add_argument("--output_confidence_pdf", default="heatmaps_confidence.pdf", type=str, help="Output PDF file for confidence heatmap")
parser.add_argument("--output_txt", default="best_toxic_tokens.txt", type=str, help="Output TXT file for toxic token dictionary")
parser.add_argument("--output_toxic_npy", default="best_average_toxic_counts.npy", type=str, help="Output Numpy file for average toxic counts")
parser.add_argument("--output_confidence_npy", default="lens_confidence.npy", type=str, help="Output Numpy file for lens confidence heatmap")
args = parser.parse_args()

# Ensure output folder exists
if not os.path.exists(args.output_folder):
    os.makedirs(args.output_folder)

# Build full output file paths
output_toxic_pdf_path = os.path.join(args.output_folder, args.output_toxic_pdf)
output_confidence_pdf_path = os.path.join(args.output_folder, args.output_confidence_pdf)
output_txt_path = os.path.join(args.output_folder, args.output_txt)
output_toxic_npy_path = os.path.join(args.output_folder, args.output_toxic_npy)
output_confidence_npy_path = os.path.join(args.output_folder, args.output_confidence_npy)

# Device setup
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if args.cpu:
    device = "cpu"

# Load model, tokenizer, and toxic words
tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, config=config)
toxic_words = load_toxic_words(args.toxic_dict_path)

# Load dataset and filter toxic prompts
dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", split="train")
toxic_prompts = dataset.filter(lambda example: example['label'] == 1)['comment_text'][:16]

# Load the best attention lenses from the folder
lense_file_paths = load_best_lens_checkpoints(args.lense_folder)
# Ensure the list is sorted by layer order if necessary
attn_lenses = [torch.load(lense_loc, map_location=torch.device(device)) for lense_loc in lense_file_paths]

# Initialize structures to hold results
total_toxic_counts = np.zeros((len(args.layer_num), args.num_attn_heads))
# Each entry now is a Counter that holds tuples (token, token_id)
common_toxic_tokens = [[Counter() for _ in range(args.num_attn_heads)] for _ in range(len(args.layer_num))]
total_confidence = np.zeros((len(args.layer_num), args.num_attn_heads))

# Process all toxic prompts
for prompt in tqdm(toxic_prompts, desc="Processing Toxic Prompts"):
    interpret_prompt(prompt, attn_lenses, common_toxic_tokens, total_toxic_counts, total_confidence, tokenizer, toxic_words, device, args.num_attn_heads, args.k_tokens)

# Calculate averages
average_toxic_counts = total_toxic_counts / len(toxic_prompts)
average_confidence = total_confidence / len(toxic_prompts)

# Save most common toxic tokens along with their token ids to a text file
with open(output_txt_path, 'w') as f:
    for layer_num, layer_common_toxic in enumerate(common_toxic_tokens):
        f.write(f"\nLayer {layer_num}:\n")
        for head, token_counter in enumerate(layer_common_toxic):
            most_common_tokens = token_counter.most_common(10)
            f.write(f"  Head {head}:\n")
            for (token_str, token_id), count in most_common_tokens:
                f.write(f"    Token: {token_str}, Count: {count}, ID: {token_id}\n")

# Create and save heatmaps:
# Heatmap for average toxic tokens per head
create_and_save_heatmap(average_toxic_counts, args.layer_num, output_toxic_pdf_path, 'Average Toxic Tokens per Head per Layer')
# Heatmap for average confidence per head
create_and_save_heatmap(average_confidence, args.layer_num, output_confidence_pdf_path, 'Average Confidence per Head per Layer')

np.save(output_toxic_npy_path, average_toxic_counts)
np.save(output_confidence_npy_path, average_confidence)