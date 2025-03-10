#!/bin/bash

# Path to your Python script
SCRIPT_PATH="find_toxic_heads.py"

# Model path (if needed)
MODEL_PATH="gpt2"

# Path to the folder containing subfolders (each with .ckpt files)
LENS_FOLDER="/grand/SuperBERT/pettyjohnjn/LoraLens/Full/ckpt_"

# Path to toxic dictionary file
TOXIC_DICT_PATH="toxic_dictionary.txt"

# Number of top token predictions
K_TOKENS=50

# Whether to force CPU usage (set to False if you want to use GPU)
FORCE_CPU=False

# Additional output file arguments
OUTPUT_FOLDER="Rank_32_Full_Outputs"
OUTPUT_TOXIC_PDF="heatmaps_toxic.pdf"
OUTPUT_CONFIDENCE_PDF="heatmaps_confidence.pdf"
OUTPUT_TXT="toxic_tokens.txt"
OUTPUT_TOXIC_NPY="average_toxic_counts.npy"
OUTPUT_CONFIDENCE_NPY="lens_confidence.npy"

# Execute the Python script with the specified arguments
python $SCRIPT_PATH \
  --model $MODEL_PATH \
  --lense_folder $LENS_FOLDER \
  --k_tokens $K_TOKENS \
  --cpu $FORCE_CPU \
  --toxic_dict_path $TOXIC_DICT_PATH \
  --output_folder $OUTPUT_FOLDER \
  --output_toxic_pdf $OUTPUT_TOXIC_PDF \
  --output_confidence_pdf $OUTPUT_CONFIDENCE_PDF \
  --output_txt $OUTPUT_TXT \
  --output_toxic_npy $OUTPUT_TOXIC_NPY \
  --output_confidence_npy $OUTPUT_CONFIDENCE_NPY \