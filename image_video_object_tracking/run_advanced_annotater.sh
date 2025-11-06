#!/bin/bash

PCD_DIR="/home/container_user/wheelchair2/src/image_video_object_tracking/pcd"
IMAGE_DIR="/home/container_user/wheelchair2/src/image_video_object_tracking/equirectangular"
CONFIG="/home/container_user/wheelchair2/src/config/config.yml"
OUTPUT_DIR="my_results_v6"

python3 advanced_annotater_v3.py \
    --data "$IMAGE_DIR" \
    --pcd_dir "$PCD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG"