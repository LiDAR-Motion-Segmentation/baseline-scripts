#!/bin/bash

# Set base paths
BAG_DIR="HK_MEMS/bags"
OUTPUT_DIR="HK_MEMS/processed"

# Camera & LiDAR topics (same for all)
CAMERA_TOPIC="/camera/color/image_raw/compressed"
CAMERA_INFO_TOPIC="/camera/color/camera_info"
LIDAR_TOPIC="/ouster/points"

# Loop through bag files (ignoring *.orig.bag)
for bag_path in "$BAG_DIR"/*.bag; do
  bag_file=$(basename "$bag_path")
  
  # Skip .orig.bag files
  if [[ "$bag_file" == *.orig.bag ]]; then
    continue
  fi

  # Remove .bag extension for output dir name
  bag_name="${bag_file%.bag}"
  echo "Processing: $bag_file"

  python3 process_rosbag.py \
    --bag-file "$BAG_DIR/$bag_file" \
    --camera-topic "$CAMERA_TOPIC" \
    --camera-info-topic "$CAMERA_INFO_TOPIC" \
    --lidar-topic "$LIDAR_TOPIC" \
    --output-dir "$OUTPUT_DIR/$bag_name"

  echo "Finished: $bag_file"
  echo "---------------------------------------------"
done

echo "✅ All bags processed."
