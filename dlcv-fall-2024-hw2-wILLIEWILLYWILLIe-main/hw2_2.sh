#!/bin/bash

# Arguments passed to the script
input_noise_dir=$1   # Path to the predefined noises
output_images_dir=$2 # Path to save generated images
model_weight=$3      # Path to the pretrained model

# Run the Python script to generate images
python3 hw2_2.py --input_noise_dir "$input_noise_dir" --output_images_dir "$output_images_dir" --model_weight "$model_weight"

# python3 hw2_2.py --input_noise_dir ./hw2_data/face/noise --output_images_dir ./output_folder/ddim --model_weight ./hw2_data/face/UNet.pt
# bash hw2_2.sh ./hw2_data/face/noise ./output_folder/ddim ./hw2_data/face/UNet.pt