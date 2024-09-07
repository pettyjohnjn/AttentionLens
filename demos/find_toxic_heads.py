import sys
sys.path.append("..")
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

# Create and save heatmap
def create_and_save_heatmap(all_toxic_counts, layer_nums, pdf_filename):
    fig, ax = plt.subplots()
    cax = ax.matshow(all_toxic_counts, cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(range(len(all_toxic_counts[0])))
    ax.set_yticks(range(len(all_toxic_counts)))
    ax.set_xticklabels(range(len(all_toxic_counts[0])))
    ax.set_yticklabels([f'Layer {layer_num}' for layer_num in layer_nums])
    plt.xlabel('Attention Head')
    plt.ylabel('Layer')
    plt.title('Average Toxic Tokens per Head per Layer')

    plt.tight_layout()
    with PdfPages(pdf_filename) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

# Process each prompt through the model and extract toxic tokens from attention heads
def interpret_prompt(prompt, attn_lenses, common_toxic_tokens, total_toxic_counts, tokenizer, toxic_words, device, num_attn_heads, k_tokens):
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
                topk_token_preds = torch.topk(projected, k_tokens).indices.cpu().numpy().tolist()
                projected_tokens = tokenizer.batch_decode(topk_token_preds)

                toxic_tokens_found = [token for token in projected_tokens if token.strip().lower() in toxic_words]
                toxic_count = len(toxic_tokens_found)

                total_toxic_counts[layer_num][head] += toxic_count
                common_toxic_tokens[layer_num][head].update(toxic_tokens_found)

# Set up user arguments
parser = argparse.ArgumentParser()

parser.add_argument("--model", default="gpt2", type=str)
parser.add_argument("--lense_loc", nargs='+', default=["/path/to/attnlens-layer-0.ckpt"], type=str)
parser.add_argument("--lens", default="gpt2", choices=["gpt2"], type=str)
parser.add_argument("--layer_num", nargs='+', default=list(range(12)), type=int)
parser.add_argument("--num_attn_heads", default=12, choices=[12, 20], type=int)
parser.add_argument("--k_tokens", default=50, type=int)
parser.add_argument("--cpu", default=True, type=bool)
parser.add_argument("--toxic_dict_path", default="toxic_dictionary.txt", type=str)
parser.add_argument("--output_pdf", default="heatmaps.pdf", type=str)
parser.add_argument("--output_txt", default="toxic_tokens.txt", type=str)

args = parser.parse_args()

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

# Load attention lenses once for all layers
attn_lenses = [torch.load(lense_loc, map_location=torch.device(device)) for lense_loc in args.lense_loc]

# Initialize structures to hold results
total_toxic_counts = np.zeros((len(args.layer_num), args.num_attn_heads))
common_toxic_tokens = [[Counter() for _ in range(args.num_attn_heads)] for _ in range(len(args.layer_num))]

# Process all toxic prompts
for prompt in tqdm(toxic_prompts, desc="Processing Toxic Prompts"):
    interpret_prompt(prompt, attn_lenses, common_toxic_tokens, total_toxic_counts, tokenizer, toxic_words, device, args.num_attn_heads, args.k_tokens)

# Calculate average toxic counts
average_toxic_counts = total_toxic_counts / len(toxic_prompts)

# Save most common toxic tokens to a text file
with open(args.output_txt, 'w') as f:
    for layer_num, layer_common_toxic in enumerate(common_toxic_tokens):
        f.write(f"\nLayer {layer_num}:\n")
        for head, token_counter in enumerate(layer_common_toxic):
            most_common_tokens = token_counter.most_common(10)
            f.write(f"  Head {head}: {most_common_tokens}\n")

# Create and save heatmap
create_and_save_heatmap(average_toxic_counts, args.layer_num, args.output_pdf)