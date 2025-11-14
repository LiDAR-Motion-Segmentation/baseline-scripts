#!/bin/bash

PCD_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/pcd"
CAM1_IMAGE_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera1"
CAM2_IMAGE_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera2"
CAM3_IMAGE_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera3"
CAM4_IMAGE_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera4"
CAM5_IMAGE_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera5"
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
CAM1_LABEL="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam1/labels_txt"
CAM2_LABEL="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam2/labels_txt"
CAM3_LABEL="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam3/labels_txt"
CAM4_LABEL="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam4/labels_txt"
CAM5_LABEL="/scratch/soumo_roy/baseline-scripts/image_video_object_tracking/my_results_cam5/labels_txt"

python3 sam_backproject_multicam.py \
    --camera camera1 "$CAM1_IMAGE_DIR" "$CAM1_INTRINSICS_DIR" "$CAM1_EXTRISICS_DIR" \
    --camera camera2 "$CAM2_IMAGE_DIR" "$CAM2_INTRINSICS_DIR" "$CAM2_EXTRISICS_DIR" \
    --camera camera3 "$CAM3_IMAGE_DIR" "$CAM3_INTRINSICS_DIR" "$CAM3_EXTRISICS_DIR" \
    --camera camera4 "$CAM4_IMAGE_DIR" "$CAM4_INTRINSICS_DIR" "$CAM4_EXTRISICS_DIR" \
    --camera camera5 "$CAM5_IMAGE_DIR" "$CAM5_INTRINSICS_DIR" "$CAM5_EXTRISICS_DIR" \
    --pcd_dir "$PCD_DIR" \
    --output_dir multicam_output_v2 \
    --labels camera1:"$CAM1_LABEL" camera2:"$CAM2_LABEL" camera3:"$CAM3_LABEL" camera4:"$CAM4_LABEL" camera5:"$CAM5_LABEL"