#!/bin/bash
CAM1_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera1"
CAM2_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera2"
CAM3_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera3"
CAM4_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera4"
CAM5_DIR="/scratch/soumo_roy/baseline-scripts/rosbag_processing_script/data_extraction/camera5"
GRDINO_CONFIG="/scratch/soumo_roy/baseline-scripts/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GRDINO_CHECKPOINT="/scratch/soumo_roy/baseline-scripts/weights/groundingdino_swint_ogc.pth"
SAM_CHECKPOINT="/scratch/soumo_roy/baseline-scripts/weights/sam_hq_vit_l.pth"

python3 grounding_dino_sam2_multicam.py \
  --camera_dirs camera1:"$CAM1_DIR" camera2:"$CAM2_DIR" camera3:"$CAM3_DIR" camera4:"$CAM4_DIR" camera5:"$CAM5_DIR" \
  --output_dir ./DINO_detection_output \
  --grounding_dino_config  "$GRDINO_CONFIG"\
  --grounding_dino_checkpoint "$GRDINO_CHECKPOINT" \
  --sam_checkpoint "$SAM_CHECKPOINT" \
  --text_prompt "person. people."
