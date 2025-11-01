from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import supervision as sv
import math
import yaml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from segment_anything_hq import SamPredictor, sam_model_registry
import argparse
import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import ros2_numpy
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class EnvironmentSetting:
    device: torch.device
    paths: Dict[str, Path]
    lidar_to_cam_matrix: np.ndarray


@dataclass
class Models:
    detection_model: AutoDetectionModel
    sam_predictor: SamPredictor


@dataclass
class Tools:
    tracker: sv.ByteTrack
    box_annotater: sv.BoxAnnotator
    mask_annotater: sv.MaskAnnotator
    label_annotater: sv.LabelAnnotator


@dataclass
class FrameData:
    frame: np.ndarray
    points_3d_lidar: np.ndarray
    points_2d_image: np.ndarray
    detections: sv.Detections


@dataclass
class ObjectData:
    json_obj: Dict[str, Any]
    label: str
    yolo_txt_line: Optional[str]


def get_center_point(bbox: np.ndarray) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def setup_environment(
    config: Dict[str, Any], data_dir: str, pcd_dir: str, output_dir: Optional[str]
) -> EnvironmentSetting:

    print("Executing setting up environment code")

    device = torch.device(
        config["models"]["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_path = Path(data_dir)
    if not output_dir:
        output_dir_path = data_path.parent / f"{data_path.stem}_advanced_annotations"
    else:
        output_dir_path = Path(output_dir)

    paths = {
        "data": data_path,
        "pcd": Path(pcd_dir),
        "output": output_dir_path,
        "labels_txt": output_dir_path / "labels_txt",
        "labels_json": output_dir_path / "labels_json",
        "visualizations": output_dir_path / "visualizations",
    }
    for path in paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)

    # ros2_numpy version
    extr = config["calib"]["extrinsics"]
    t = extr["translation"]
    q = extr["rotation"]
    translation_m = ros2_numpy.geometry.transformations.translation_matrix(t)
    rotation_m = ros2_numpy.geometry.transformations.quaternion_matrix(q)
    T_lidar_camera = np.dot(translation_m, rotation_m)
    lidar_to_cam_matrix = np.linalg.inv(T_lidar_camera)

    return EnvironmentSetting(
        device=device, paths=paths, lidar_to_cam_matrix=lidar_to_cam_matrix
    )


def load_models(config: Dict[str, Any], device: torch.device) -> Models:
    print("lodaing models")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=config["paths"]["yolo_model"],
        confidence_threshold=config["detection"]["confidence_threshold"],
        device=device,
    )

    sam = sam_model_registry[config["models"]["sam_model_type"]]()
    state_dict = torch.load(config["paths"]["sam_checkpoint"], map_location="cpu")
    sam.load_state_dict(state_dict)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    return Models(detection_model=detection_model, sam_predictor=sam_predictor)


def initialize_tools(config: Dict[str, Any]) -> Tools:
    print("initialized tools")

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)

    label_annotater = sv.LabelAnnotator(text_scale=0.6, text_color=sv.Color.BLACK)

    return Tools(
        tracker=tracker,
        box_annotater=box_annotator,
        mask_annotater=mask_annotator,
        label_annotater=label_annotater,
    )


def load_frame_data(
    image_path: Path, pcd_base_path: Path
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    frame = cv2.imread(str(image_path))
    pcd_file = pcd_base_path / f"{image_path.stem}.pcd"

    if not pcd_file.exists():
        print(f"Warning: Point cloud not found for {image_path.name}. Skipping.")
        return None

    pcd = o3d.io.read_point_cloud(str(pcd_file))
    points_3d_lidar = np.asarray(pcd.points)

    if frame is None:
        print(f"Warning: Could not read image {image_path.name}. Skipping.")
        return None

    return frame, points_3d_lidar


def run_2d_pipeline(
    frame: np.ndarray, models: Models, tools: Tools, config: Dict[str, Any]
) -> sv.Detections:
    sahi_result = get_sliced_prediction(
        frame,
        models.detection_model,
        slice_height=config["sahi"]["slice_height"],
        slice_width=config["sahi"]["slice_width"],
        overlap_height_ratio=config["sahi"]["overlap_ratio"],
        overlap_width_ratio=config["sahi"]["overlap_ratio"],
    )

    xyxy_list, confidence_list, class_id_list = [], [], []
    if sahi_result.object_prediction_list:
        for pred in sahi_result.object_prediction_list:
            xyxy_list.append(pred.bbox.to_xyxy())
            confidence_list.append(pred.score.value)
            class_id_list.append(pred.category.id)

    detections = sv.Detections(
        xyxy=np.array(xyxy_list),  # size (0,4)
        confidence=np.array(confidence_list),  # size (0)
        class_id=np.array(class_id_list).astype(int),  # size (0)
    )

    detections = detections[detections.class_id == 0]  # filtering people
    detections = detections.with_nms(threshold=config["detection"]["nms_theshold"])
    tracked_detection = tools.tracker.update_with_detections(detections)
    tracked_detection = tracked_detection[tracked_detection.tracker_id != None]

    return tracked_detection
