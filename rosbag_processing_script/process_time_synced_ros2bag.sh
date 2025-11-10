#!/bin/bash

BAG_FILE="/scratch2/soumo_roy/nexus_mall_bags/nexus_lower_ground_2/rosbag/rosbag_0.db3"
CAMERA_TOPIC_1="/camera1/camera1/color/image_raw"
CAMERA_TOPIC_2="/camera2/camera2/color/image_raw"
CAMERA_TOPIC_3="/camera3/camera3/color/image_raw"
CAMERA_TOPIC_4="/camera4/camera4/color/image_raw"
CAMERA_TOPIC_5="/camera5/camera5/color/image_raw"
LIDAR_TOPIC="/livox/lidar"
OUTPUT_DIR="extracted_data"

python3 extract_synced_data_multi_cam.py \
    --bag_file "$BAG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --lidar_topic "$LIDAR_TOPIC" \
    --image_topics "$CAMERA_TOPIC_1 $CAMERA_TOPIC_2" 
