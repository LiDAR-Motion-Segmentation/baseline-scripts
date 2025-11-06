#!/bin/bash

PCD_DIR=""
IMAGE_DIR=""
CONFIG=""
OUTPUT_DIR=""

python3 advanced_annotater_v3.py \
    --data "$IMAGE_DIR" \
    --pcd_dir "$PCD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG"