# baseline-scripts for temporal-point-transformer
Extra utility codes used to run baselines smoothly, for JRDB dataset and semantic kitty dataset. 

## semantic kitty scripts
- the files are placed in `training_script_semantic_kitty` directory

- semantic kitty dataset structure, Download it from here [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) (including **Velodyne point clouds**, **calibration data** and **label data**).
- instructions for JRDB will be added soon and scripts are in `training_script_JRDB` directory
```
DATAROOT
└── sequences
    ├── 00
    │   ├── poses.txt
    │   ├── calib.txt
    │   ├── times.txt
    │   ├── labels
    │   └── velodyne
    |── 01-10

# sequences for training: 00-10
# sequences for validation: 08
# sequences for testing: 08
```

- activate the evironment (the environment file is environment.yml)
```
conda env create -f environment.yml
conda activate lidar_moseg
pip install e .
```
- to run and test the train dataloader and test dataloader script (move this script to the dataloader directory in codebase which ever is used)
```
python3 dataloaders/semantic_kitty.py
```
- to run the training script use (-m to be used when running in a module otherwise it not reqired also put this code in scripts directory)
```
tmux new -s training
tmux a -t training
python3 -m scripts.train_semantic_kitty.py 
```
- to use the first GPU on your system incase  it is not detected use this
```
export CUDA_VISIBLE_DEVICES=0
```
- To run the evaluation script for semantic kitty use
```
python3 -m scripts.eval_semantic_kitty --config_path <path>/config/semantic_kitty_config.yaml --checkpoint_path <path>/best-checkpoint-epoch=07-val_loss=0.00-v1.ckpt
```

- the utilities for point cloud  processing is present in `pointcloud_utils.py` and the config used for training along with the split is present in config folder in `semantic_kitty_config.yaml`

## Results
- by default, wandb logging is turned on, so if you wish to use your wandb account, please make a .env file, with your wandb api key as follows
```
WANDB_API_KEY=<YOUR-API-KEY>
```
- also change the `semantic-kitty-config.yaml` to add the details
```
logging:
  wandb:
    run_root_dir: "/scratch/<username>/temporal-point-transformer"
    project: "add your project here"
    entity: "add your entity here"
    log_model: False
    save_code: False
    group: "temporal-point"
    name: "patch64-semantic-kitty"
    resume: "never"
    log_dir: "/scratch/<username>/temporal-point-transformer/logs"
```
- Blue line is on JRDB dataset and red line is on Semantic kitty dataset
![alt text](./assets/image.png)

- In Scripts folder `eval_semantic_kitty.py` should print the output below in this way for sequence 8
```
==================================================
🧪 Test Metrics Summary
==================================================
🔸 Loss      : 0.0536
🔸 IoU       : 0.7081
🔸 Precision : 0.8105
🔸 Recall    : 0.7468
🔸 F1 Score  : 0.7655
==================================================
```

## Visualization
- use rerun to visualize the result.
```
rerun --serve & disown
python3 -m scripts.visualize_semantic_kitty --config_path <path>/config/semantic_kitty_config.yaml --checkpoint_path <path>/best-checkpoint-epoch=07-val_loss=0.00-v1.ckpt
```
- left side is the predictions and right side is the ground truth
![alt text](./assets/image-1.png)

## ROSbag processing for custom Dataset

- remember to change the directories in the bash scripts as per your directory path for seamless usage
```
# for ROS2 humble bags
./rosbag_processing_script/process_ros2_bag.sh

# for ROS1 noetic bags
./rosbag_processing_script/process_ros1_bag.sh
```
- it gives a directory with images,camera intrinsics and pointclouds seperately with synced timestamps
```
DATAROOT
└──bag name
    ├── camera1_images      # (.png format) (format : 0000000000000000000.png)
    ├── camera1_intrinsics  # (.npz format) (format : D,K,height,width)
    ├── camera2_images      # (.png format) (format : 0000000000000000000.png)
    ├── camera2_intrinsics  # (.npz format) (format : D,K,height,width)
    ├── lidar               # (.pcd format) (format : 0000000000000000000.pcd)
```

##  Custom Data Annotation
1) Multi-Camera Support: Leverages both front and back cameras

2) Synchronized Data: Properly handles temporal synchronization

3) Semi-Automatic: YOLO pre-annotation reduces manual work

4) Frustum Projection: Accurate 3D labeling from 2D detections

5) Visualization: Comprehensive visualizations for quality control
```
# Step 1: Analyze data
python custom_analyzer.py /path/to/DATAROOT/my_bag

# Step 2: Generate annotations (with YOLOv8 only)
pip install ultralytics
python rosbag_processing_script/rosbag_custom_annotator.py my_bag_sync_map.json --output my_bag_annotations

OR
# Step 2: Generate annotations (with YOLOv8 + SAM2 )
pip install git+https://github.com/facebookresearch/segment-anything-2.git
python3 rosbag_processing_script/enhanced_annotater.py my_bag_sync_map.json --output annotation_sam2

# Step 3: Convert to KITTI
python convert_to_kitti.py my_bag_annotations --output my_kitti_dataset
```

Custom Data structure after running the code above with YOLO
```
output_directory/
├── labels/                    # .label files
├── visualizations/            # .pcd files (labeled point clouds)
├── visualization_camera_1/    # .png files from camera 1
│   ├── 0000000000000000001_camera1.png
│   ├── 0000000000000000002_camera1.png
│   └── ...
├── visualization_camera_2/    # .png files from camera 2
│   ├── 0000000000000000001_camera2.png
│   ├── 0000000000000000002_camera2.png
│   └── ...
└── annotation_summary.json
```
Custom Data structure after running the code with YOLO+SAM2
```
annotations_YOLO_SAM2/
├── labels/                          # .label files for point cloud labels
├── visualization/                   # Labeled point clouds (.pcd)
├── visualization_camera_1/          # Camera 1 with bounding boxes + mask overlays
├── visualization_camera_2/          # Camera 2 with bounding boxes + mask overlays
├── segmentation_masks/
│   ├── camera1/                     # Individual SAM2 masks for camera1
│   ├── camera2/                     # Individual SAM2 masks for camera2
│   ├── TIMESTAMP_camera1_combined.png
│   └── TIMESTAMP_camera2_combined.png
├── annotation_summary.json         # Complete summary (JSON-safe)
└── annotation_summary_simplified.json  # Fallback if main JSON fails

```

## Acknowledgment
- I have used [temporal-point-transformer](https://github.com/LiDAR-Motion-Segmentation/temporal-point-transformer) model to train and evaluate on.

## Citation
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```