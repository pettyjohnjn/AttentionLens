#!/bin/bash

# Path to the new Python visualization script
SCRIPT_PATH="visualize_heads.py"

# Model path (e.g., "gpt2")
MODEL_PATH="gpt2"

# Path to the folder containing subfolders (each with .ckpt files for each layer)
LENS_FOLDER="/grand/SuperBERT/pettyjohnjn/LoraLens_Pile2/extracted_checkpoint/gpt2/R02"

# Layer index to visualize (0-indexed)
LAYER_INDEX=8

# Attention head index to visualize (0-indexed)
HEAD_INDEX=2

# Maximum number of tokens to sample
MAX_TOKENS=50768

# Number of clusters for K-means clustering
N_CLUSTERS=20

# Output HTML file name
OUTPUT_HTML="token_clusters.html"

# Comma-separated token ids to highlight (enclosed in quotes)
HIGHLIGHT_TOKEN_IDS="7510,5089,30998,18824,46733,32574,20654,31699,41356"

# Whether to force CPU usage (pass the flag if desired; remove if using GPU)
FORCE_CPU="--cpu"

python $SCRIPT_PATH \
  --model $MODEL_PATH \
  --lens_folder $LENS_FOLDER \
  --layer $LAYER_INDEX \
  --head $HEAD_INDEX \
  --max_tokens $MAX_TOKENS \
  --n_clusters $N_CLUSTERS \
  --output_html $OUTPUT_HTML \
  --highlight_token_ids "$HIGHLIGHT_TOKEN_IDS" \