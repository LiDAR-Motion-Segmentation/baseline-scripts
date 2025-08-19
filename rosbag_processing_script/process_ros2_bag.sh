#!/bin/bash

BAG=aug5-group-people-walkingout-with-door-open-no-ego-motion

# === Configuration ===
BAG_FILE="/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/soumo/aug5/group-people-walkingout-with-door-open-no-ego-motion/rosbag/rosbag_0.db3"
CAMERA_TOPIC_1="/camera1/camera1/color/image_raw"
CAMERA_TOPIC_2="/camera2/camera2/color/image_raw"
CAMERA_INFO_TOPIC_1="/camera1/camera1/color/camera_info"
CAMERA_INFO_TOPIC_2="/camera2/camera2/color/camera_info"
LIDAR_TOPIC="/livox/lidar"
OUTPUT_DIR="/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/soumo/aug5/processed_bags/$BAG"
SYNC_TOL=0.1

# === Run the script ===
python3 process_ros2_bag.py \
    --bag-file "$BAG_FILE" \
    --camera-topic "$CAMERA_TOPIC_1" \
    --camera-topic-2 "$CAMERA_TOPIC_2" \
    --camera-info-topic "$CAMERA_INFO_TOPIC_1" \
    --camera-info-topic-2 "$CAMERA_INFO_TOPIC_2" \
    --lidar-topic "$LIDAR_TOPIC" \
    --output-dir "$OUTPUT_DIR" \
    --sync-tol "$SYNC_TOL" \
    --dual-camera