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
    lidar_to_cam_matrix: np.array


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
    for path in path.values():
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

