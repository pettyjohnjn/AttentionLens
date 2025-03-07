#!/usr/bin/env python
import sys
sys.path.append("..")
import argparse
import os
import re
import torch
import numpy as np
import random
import pandas as pd
import umap.umap_ as umap
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from sklearn.cluster import KMeans
import plotly.graph_objects as go

def load_best_lens_checkpoints(lens_folder):
    best_checkpoints = []
    for subfolder in sorted(os.listdir(lens_folder)):
        subfolder_path = os.path.join(lens_folder, subfolder)
        if os.path.isdir(subfolder_path):
            ckpt_files = [f for f in os.listdir(subfolder_path) if f.endswith('.ckpt')]
            best_loss = float('inf')
            best_ckpt = None
            for ckpt in ckpt_files:
                match = re.search(r"train_loss=([\d\.]+)", ckpt)
                if match:
                    loss = float(match.group(1).rstrip('.'))
                    if loss < best_loss:
                        best_loss = loss
                        best_ckpt = ckpt
            if best_ckpt is not None:
                best_checkpoints.append(os.path.join(subfolder_path, best_ckpt))
    return best_checkpoints

def visualize_token_clusters(args):
    # Setup device
    device = "cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu"
    
    # Load model, tokenizer, and config
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config)
    model.to(device)
    
    # Load lens checkpoint for the specified layer.
    lens_file_paths = load_best_lens_checkpoints(args.lens_folder)
    if len(lens_file_paths) == 0:
        print("No lens checkpoints found in the provided folder.")
        return
    layer_idx = args.layer
    try:
        lens_path = lens_file_paths[layer_idx]
    except IndexError:
        print(f"Layer index {layer_idx} out of range. Found only {len(lens_file_paths)} layers.")
        return
    lens = torch.load(lens_path, map_location=device)
    
    # Extract the linear transformation for the specified attention head.
    head_idx = args.head
    linear_layer = lens.linears[head_idx]
    linear_layer.eval()
    
    # Get weight and bias from the linear transformation.
    weight = linear_layer.weight.detach()  # shape: (vocab_size, residual_dim)
    if linear_layer.bias is not None:
        bias = linear_layer.bias.detach()
    else:
        bias = torch.zeros(weight.size(0), device=weight.device)
    
    # Compute the pseudo-inverse of the weight matrix.
    pinv_weight = torch.linalg.pinv(weight)
    
    # Get the full vocabulary (sorted by token id).
    vocab = tokenizer.get_vocab()  # dict mapping token -> id
    all_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Randomly sample tokens from the full vocabulary.
    if args.max_tokens < len(all_vocab):
        sampled_vocab = random.sample(all_vocab, args.max_tokens)
    else:
        sampled_vocab = all_vocab.copy()
    
    # Ensure that highlighted tokens are included.
    highlight_ids = set()
    if args.highlight_token_ids:
        try:
            highlight_ids = set(map(int, args.highlight_token_ids.split(',')))
        except Exception as e:
            print("Error parsing highlight_token_ids:", e)
            highlight_ids = set()
    highlight_tokens = [item for item in all_vocab if item[1] in highlight_ids]
    sample_ids = {item[1] for item in sampled_vocab}
    for token_item in highlight_tokens:
        if token_item[1] not in sample_ids:
            sampled_vocab.append(token_item)
    
    # Create lists for tokens and their corresponding ids.
    tokens = [token for token, idx in sampled_vocab]
    token_ids = [idx for token, idx in sampled_vocab]
    
    vocab_size = weight.size(0)
    residual_reps = []
    for idx in token_ids:
        one_hot = torch.zeros(vocab_size, device=weight.device)
        one_hot[idx] = 1.0
        residual = pinv_weight.matmul(one_hot - bias)
        residual_reps.append(residual.cpu().numpy())
    residual_reps = np.array(residual_reps)
    
    # Dimensionality reduction with UMAP.
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(residual_reps)
    
    # Cluster tokens with K-means.
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embedding)
    
    # Create a DataFrame for Plotly.
    df = pd.DataFrame({
        "UMAP1": embedding[:, 0],
        "UMAP2": embedding[:, 1],
        "Cluster": clusters,
        "Token": tokens,
        "TokenID": token_ids,
    })
    df["Highlighted"] = df["TokenID"].apply(lambda x: x in highlight_ids)
    
    # Create the interactive figure.
    fig = go.Figure()
    
    # Use Scattergl for the main set of tokens (for efficient rendering of many points).
    fig.add_trace(go.Scattergl(
        x=df["UMAP1"],
        y=df["UMAP2"],
        mode='markers',
        marker=dict(
            color=df["Cluster"],
            colorscale='Spectral',
            size=5,
            opacity=0.8
        ),
        text=df["Token"],
        hovertemplate="Token: %{text}<br>TokenID: %{customdata}",
        customdata=df["TokenID"],
        name="Tokens"
    ))
    
    # Add a separate trace for highlighted tokens that always appears on top.
    if df["Highlighted"].any():
        highlight_df = df[df["Highlighted"]]
        fig.add_trace(go.Scatter(
            x=highlight_df["UMAP1"],
            y=highlight_df["UMAP2"],
            mode="markers+text",
            marker=dict(symbol="x", color="red", size=12),
            text=highlight_df["Token"],
            textposition="top center",
            name="Highlighted Tokens",
            hovertemplate="Token: %{text}<br>TokenID: %{customdata}",
            customdata=highlight_df["TokenID"],
        ))
    
    # Update layout for responsiveness.
    fig.update_layout(
        title=f"Token Clusters for Layer {args.layer}, Head {args.head}",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        autosize=True,
        hovermode="closest",
    )
    
    # Save the interactive plot as an HTML file.
    fig.write_html(args.output_html, full_html=True, include_plotlyjs="cdn")
    print(f"Interactive visualization saved to {args.output_html}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Interactive visualization of token clusters for a given attention head."
    )
    parser.add_argument("--model", default="gpt2", type=str, help="Model name (e.g., gpt2)")
    parser.add_argument("--lens_folder", default="/path/to/lens_folder", type=str,
                        help="Folder containing subfolders of lens checkpoints for each layer")
    parser.add_argument("--layer", default=0, type=int, help="Layer index to visualize")
    parser.add_argument("--head", default=0, type=int, help="Attention head index to visualize")
    parser.add_argument("--max_tokens", default=1000, type=int, help="Maximum number of tokens to sample")
    parser.add_argument("--n_clusters", default=10, type=int, help="Number of clusters for K-means")
    parser.add_argument("--output_html", default="token_clusters.html", type=str, help="Output HTML file name")
    parser.add_argument("--cpu", action='store_true', help="Force CPU usage even if CUDA is available")
    parser.add_argument("--highlight_token_ids", default="", type=str,
                        help="Comma-separated list of token ids to highlight in the visualization")
    
    args = parser.parse_args()
    visualize_token_clusters(args)