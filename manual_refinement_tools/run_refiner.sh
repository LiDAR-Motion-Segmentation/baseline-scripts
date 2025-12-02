#!/bin/bash

CAM1_IMAGE_DIR="/home/soumoroy/Downloads/MCMPT_output_nice_op/camera1"
CAM1_JSON_DIR="/home/soumoroy/Downloads/MCMPT_output_nice_op/cam_1/json"
CAM1_OUTPUT_DIR="/home/soumoroy/Downloads/MCMPT_output_nice_op/final_dataset/cam_1"
CONFIG="/home/soumoroy/baseline-scripts/config/ui_theme.yaml"

python3 manual_refiner_v2.py \
    --img_dir "$CAM1_IMAGE_DIR" \
    --json_dir "$CAM1_JSON_DIR" \
    --out_dir "$CAM1_OUTPUT_DIR" \
    --config "$CONFIG"