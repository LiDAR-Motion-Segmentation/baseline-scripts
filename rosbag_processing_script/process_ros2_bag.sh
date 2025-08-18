#!/bin/bash

BAG=sss_himalaya_rosbag

# === Configuration ===
BAG_FILE="WC/bags/sss_himlaya_rosbag/rosbag_0.db3"
CAMERA_TOPIC_1="/camera/camera/color/image_raw"
#CAMERA_TOPIC_2= "/camera2/camera2/color/image_raw"
CAMERA_INFO_TOPIC_1="/camera/camera/color/camera_info"
#CAMERA_INFO_TOPIC_2="/camera2/camera2/color/camera_info"
LIDAR_TOPIC="/livox/lidar"
OUTPUT_DIR="WC/processed/$BAG"
SYNC_TOL=0.1

# === Run the script ===
python3 process_ros2_bag.py \
    --bag-file "$BAG_FILE" \
    --camera-topic "$CAMERA_TOPIC_1" \
    --camera-info-topic "$CAMERA_INFO_TOPIC_1" \
    --lidar-topic "$LIDAR_TOPIC" \
    --output-dir "$OUTPUT_DIR" \
    --sync-tol "$SYNC_TOL"