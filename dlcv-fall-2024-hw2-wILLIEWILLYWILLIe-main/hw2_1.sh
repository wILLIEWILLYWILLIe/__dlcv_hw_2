#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: bash hw2_1.sh <output_directory>"
    exit 1
fi

# Assign the first argument to the output directory variable
OUTPUT_DIR=$1

# Run the Python script with the provided output directory
python3 hw2_1.py --outputDir "$OUTPUT_DIR"

# # bash hw2_1.sh ./output_folder
# python3 digit_classifier.py --folder ./output_folder --checkpoint ./Classifier.pth

