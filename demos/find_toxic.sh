#!/bin/bash

# Path to your Python script
SCRIPT_PATH="find_toxic_heads.py"

# Model path (if needed)
MODEL_PATH="gpt2"

# Paths to attention lenses (ensure all file paths are correct)
LENS_PATHS=(
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_0/attnlens-layer-0-epoch=00-step=1635-train_loss=1.72.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_1/attnlens-layer-1-epoch=00-step=915-train_loss=1.41.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_2/attnlens-layer-2-epoch=00-step=335-train_loss=3.46.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_3/attnlens-layer-3-epoch=00-step=605-train_loss=2.62.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_4/attnlens-layer-4-epoch=00-step=730-train_loss=3.55.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_5/attnlens-layer-5-epoch=00-step=2080-train_loss=1.01.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_6/attnlens-layer-6-epoch=00-step=1625-train_loss=1.25.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_7/attnlens-layer-7-epoch=00-step=640-train_loss=1.88.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_8/attnlens-layer-8-epoch=00-step=1080-train_loss=1.85.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_9/attnlens-layer-9-epoch=00-step=1005-train_loss=0.72.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_10/attnlens-layer-10-epoch=00-step=565-train_loss=2.41.ckpt"
    "/lus/grand/projects/SuperBERT/pettyjohnjn/Extracted_lens/gpt2/layer_11/attnlens-layer-11-epoch=00-step=490-train_loss=5.80.ckpt"
)

# Layer numbers corresponding to each lens path
LAYER_NUMS=(0 1 2 3 4 5 6 7 8 9 10 11)

# Path to toxic dictionary file
TOXIC_DICT_PATH="toxic_dictionary.txt"

# Output PDF file for heatmaps
OUTPUT_PDF="output_heatmaps.pdf"

# Number of top token predictions
K_TOKENS=50

# Whether to force CPU usage (set to False if you want to use GPU)
FORCE_CPU=False

# Execute the Python script with the specified arguments
python $SCRIPT_PATH \
  --model $MODEL_PATH \
  --lense_loc "${LENS_PATHS[@]}" \
  --layer_num "${LAYER_NUMS[@]}" \
  --k_tokens $K_TOKENS \
  --cpu $FORCE_CPU \
  --toxic_dict_path $TOXIC_DICT_PATH \
  --output_pdf $OUTPUT_PDF