#!/bin/bash

PCD_DIR="/home/container_user/wheelchair2/src/rosbag_processing_script/data_extraction/pcd/"
IMAGE_DIR="/home/container_user/wheelchair2/src/rosbag_processing_script/data_extraction/camera1/"
LABEL_DIR="/home/container_user/wheelchair2/src/image_video_object_tracking/my_results_v6/labels_txt/"
CONFIG="/home/container_user/wheelchair2/src/config/calibration.yml"
OFFSET="0"
OUTPUT_DIR="my_results_v7"

python3 sam_backproject_v3.py \
    --pcd_dir "$PCD_DIR" \
    --image_dir "$IMAGE_DIR" \
    --label_dir "$LABEL_DIR" \
    --config "$CONFIG" \
    --offset "$OFFSET" \
    --output_dir "$OUTPUT_DIR"