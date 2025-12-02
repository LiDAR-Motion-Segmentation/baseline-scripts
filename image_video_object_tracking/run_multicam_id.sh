#!/bin/bash

PCD_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/pcd"
CAM1_INTRINSICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera1_intrinsics.txt"
CAM2_INTRINSICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera2_intrinsics.txt"
CAM3_INTRINSICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera3_intrinsics.txt"
CAM4_INTRINSICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera4_intrinsics.txt"
CAM5_INTRINSICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera5_intrinsics.txt"
CAM1_EXTRISICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera1_extrinsics.txt"
CAM2_EXTRISICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera2_extrinsics.txt"
CAM3_EXTRISICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera3_extrinsics.txt"
CAM4_EXTRISICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera4_extrinsics.txt"
CAM5_EXTRISICS_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera5_extrinsics.txt"
CAM1_JSON_DIR="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam1/json"
CAM2_JSON_DIR="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam2/json"
CAM3_JSON_DIR="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam3/json"
CAM4_JSON_DIR="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam4/json"
CAM5_JSON_DIR="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam5/json"

python generate_3d_annotations_with_id.py \
  --pcd_dir "$PCD_DIR" \
  --out_dir ./3d_labels_with_id \
  --cam cam1 "$CAM1_INTRINSICS_DIR" "$CAM1_EXTRISICS_DIR" "$CAM1_JSON_DIR" \
  --cam cam2 "$CAM2_INTRINSICS_DIR" "$CAM2_EXTRISICS_DIR" "$CAM2_JSON_DIR" \
  --cam cam3 "$CAM3_INTRINSICS_DIR" "$CAM3_EXTRISICS_DIR" "$CAM3_JSON_DIR" \
  --cam cam4 "$CAM4_INTRINSICS_DIR" "$CAM4_EXTRISICS_DIR" "$CAM4_JSON_DIR" \
  --cam cam5 "$CAM5_INTRINSICS_DIR" "$CAM5_EXTRISICS_DIR" "$CAM5_JSON_DIR"