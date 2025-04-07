#!/bin/bash

# hw2_3.sh
# Usage: bash hw2_3.sh <input_json> <output_folder>

JSON_FILE=$1
OUTPUT_FOLDER=$2
MODEL_CKPT=$3

python3 hw2_3.py $JSON_FILE $OUTPUT_FOLDER $MODEL_CKPT

# python3 evaluation/grade_hw2_3.py --json_path ./hw2_data/textual_inversion/input.json --input_dir ./hw2_data/textual_inversion --output_dir ./output_folder
# bash hw2_3.sh ./hw2_data/textual_inversion/input.json ./output_folder ./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt
