#!/bin/bash

BAG=sep11-going-towards-ihub

# === Configuration ===
BAG_FILE="/scratch2/soumo_roy/Motion-segementation-rosbags/sep/going-towards-ihub/rosbag/rosbag_0.db3"
CAMERA_TOPIC_1="/camera1/camera1/color/image_raw"
CAMERA_TOPIC_2="/camera2/camera2/color/image_raw"
CAMERA_INFO_TOPIC_1="/camera1/camera1/color/camera_info"
CAMERA_INFO_TOPIC_2="/camera2/camera2/color/camera_info"
CAMERA_TOPIC_3="/dual_fisheye/image"
LIDAR_TOPIC="/livox/lidar"
OUTPUT_DIR="/scratch2/soumo_roy/Motion-segementation-rosbags/soumo/sep11/processed_bags/$BAG"
SYNC_TOL=0.1

# === Run the script ===
# try this if you run out of memory
# ulimit -Sv unlimited && nice -n 19 
python3 rosbag_processing_script/process_ros2_bag.py \
    --bag-file "$BAG_FILE" \
    --camera-topic "$CAMERA_TOPIC_1" \
    --camera-topic-2 "$CAMERA_TOPIC_2" \
    --camera-topic-3 "$CAMERA_TOPIC_3" \
    --camera-info-topic "$CAMERA_INFO_TOPIC_1" \
    --camera-info-topic-2 "$CAMERA_INFO_TOPIC_2" \
    --lidar-topic "$LIDAR_TOPIC" \
    --output-dir "$OUTPUT_DIR" \
    --sync-tol "$SYNC_TOL" \
    --multi-camera