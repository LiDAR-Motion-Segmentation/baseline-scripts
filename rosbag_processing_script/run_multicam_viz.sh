#!/bin/bash

CAM1_DIR="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_v8/sam_image"
CAM2_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera2"
CAM3_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera3"
CAM4_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera4"
CAM5_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera5"

python3 rerun_multi_cam_viewer.py \
    --cam_dirs cam0 "$CAM1_DIR" \
    --cam_dirs cam1 "$CAM2_DIR" \
    --cam_dirs cam2 "$CAM3_DIR" \
    --cam_dirs cam3 "$CAM4_DIR"\
    --cam_dirs cam4 "$CAM5_DIR"