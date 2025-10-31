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
