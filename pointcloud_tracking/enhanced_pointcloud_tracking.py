import os
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import math
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import open3d as o3d
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation as R

# Optional: Enhanced tracking and visualization
# might remove in future just trying it out
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# as of now considering the previous data that we have collected
class ObjectType(Enum):
    MOVING_PEOPLE = "moving_people"
    PEOPLE_STATIC = "people_static"
    MOVING_MOBILE_ROBOT = "moving_mobile_robot"
    STATIC_OBJECT = "static_object"
    UNKNOWN = "unknown"
    
@dataclass
class BoundingBox3D:
    center: np.ndarray # [x, y, z]
    size: np.ndarray   # [length, width, height]
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0,0.0,0.0])) # [rx, ry, rz]
    confidence: float = 1.0
    obj_id: str = ""
    obj_type: ObjectType = ObjectType.UNKNOWN
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0,0.0,0.0]))
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "obj_id": self.obj_id,
            "obj_type": self.obj_type.value,
            "psr":{
                "position": {
                    "x": float(self.center[0]),
                    "y": float(self.center[1]),
                    "z": float(self.center[2])
                },
                "rotation": {
                    "x": float(self.rotation[0]),
                    "y": float(self.rotation[1]),
                    "z": float(self.rotation[2])
                },
                "scale": {
                    "x": float(self.size[0]),
                    "y": float(self.size[1]),
                    "z": float(self.size[2])                    
                }
            },
             "metadata": {
                "confidence": float(self.confidence),
                "velocity": {
                    "x": float(self.velocity[0]),
                    "y": float(self.velocity[1]),
                    "z": float(self.velocity[2])
                },
                "timestamp": float(self.timestamp)
            }
        }
        
@dataclass
class TrackingResult:
    frame_id: int
    timestamp: float
    boxes: List[BoundingBox3D]
    pointcloud_path: str
    processing_time: float = 0.0
    detection_count: Dict[ObjectType, int] = field(default_factory=dict)
    
# history lenght needs to be checked
class EnhancedMotionAnalyser:
    def __init__(self, history_length: int = 15, motion_threshold: float = 0.1):
        self.history_length = history_length
        self.motion_threshold = motion_threshold
        self.velocity_threshold = 0.1 # m/s
        self.track_histories = defaultdict(lambda: deque(maxlen=history_length))
        self.object_characteristics = {}
        
    def update_track(self, obj_id: str, bbox: BoundingBox3D, timestamp: float):
        history_entry = {
            'positon': bbox.center.copy(),
            'size': bbox.size.copy(),
            'timestamp': timestamp,
            'confidence': bbox.confidence
        }
        self.track_histories[obj_id].append(history_entry)
        
        if obj_id not in self.object_charactertics:
            self.object_charactertics[obj_id] = {
                'avg_size': bbox.size.copy(),
                'size_variance': np.zeros(3),
                'typical_height': bbox.size[2]
            }
        else:
            char = self.object_charactertics[obj_id]
            alpha = 0.1
            char['avg_size'] = (1 - alpha) * char['avg_size'] + alpha * bbox.size
            char['size_variance'] = (1 - alpha) * char['size_variance'] + alpha * (bbox.size - char['avg_size']) ** 2
            
    def classify_object_type(self, obj_id: str) -> ObjectType:
        if obj_id not in self.track_histories:
            return ObjectType.UNKNOWN
        
        history = list(self.track_histories[obj_id])
        if len(history) < 3:
            return ObjectType.UNKNOWN
        
        char = self.object_charactertics.get(obj_id, {})
        avg_size = char.get('avg_size', np.array([0.5, 0.5, 1.7]))
        motion_stats = self._calculate_motion_statistics(history)
        
        is_human_sized = self._is_human_sized(avg_size)
        is_robot_sized = self._is_robot_sized(avg_size)
        is_moving = motion_stats['avg_velocity'] > self.motion_threshold
        is_fast_moving = motion_stats['max_velocity'] > self.velocity_threshold
        
        if is_human_sized:
            if is_moving:
                return ObjectType.MOVING_PEOPLE
            else:
                return ObjectType.PEOPLE_STATIC
        elif is_robot_sized and is_fast_moving:
            return ObjectType.MOVING_MOBILE_ROBOT
        elif not is_moving:
            return ObjectType.STATIC_OBJECT
        
    def _calculate_motion_statistics(self, history: List[Dict]) -> Dict[str, float]:
        if len(history) < 2:
            return {'avg_velocity': 0.0, 'max_velocity': 0.0, 'acceleration': 0.0}
        position = np.array((h['position'] for h in history))
        timestamp = np.array((h['timestamp'] for h in history))
        velocities = []
        for i in range(1, len(position)):
            dt = timestamp[i] - timestamp[i-1]
            if dt > 0:
                velocity = np.linalg.norm(position[i] - position[i-1]) / dt
                velocities.append(velocity)
                
        if not velocities:
            return  {'avg_velocity': 0.0, 'max_velocity': 0.0, 'acceleration': 0.0}
        
        acceleration = 0.0
        if len(history) >= 2:
            vel_changes = np.diff(velocities)
            acceleration = np.mean(np.abs(vel_changes))
            
        return {
            'avg_velocity': np.mean(velocities),
            'max_velocity': np.max(velocities),
            'acceleration': acceleration,
            'velocity_std': np.std(velocities)
        }
        
    # might need to change the values in future    
    def _is_human_sized(self, size: np.ndarray) -> bool:
        length, width, height = size
        return (0.3 <= length <= 0.8 and 
                0.3 <= width <= 0.8 and 
                1.2 <= height <= 2.2)

    def _is_robot_sized(self, size: np.ndarray) -> bool:
        length, width, height = size
        return (0.4 <= length <= 1.2 and 
                0.4 <= width <= 1.2 and 
                0.3 <= height <= 1.5)
        
class EnhancedPointPillarDetector:
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.confidence_threshold = 0.4
        self.nms_threshold = 0.5
        
        # testing some standard values might need to change in the future
        self.object_specs = {
            'human': {
                'size_range': {'length': (0.3, 0.8), 'width': (0.3, 0.8), 'height': (1.2, 2.2)},
                'height_filter': (0.5, 2.5),
                'min_points': 20
            },
            'robot': {
                'size_range': {'length': (0.4, 1.2), 'width': (0.4, 1.2), 'height': (0.3, 1.5)},
                'height_filter': (0.2, 1.8),
                'min_points': 30
            }
        }
        logger.info(f"Initialized Enhanced PointPillars detector on {device}")
        
    def detect_objects(self, pointcloud: np.ndarray) -> List[BoundingBox3D]:
        if pointcloud.shape[0] < 100:
            return []
        detections = []
        
        for obj_class, specs in self.object_specs.items():
            class_detections = self._detect_object_class(pointcloud, obj_class, specs)
            detections.extend(class_detections)
        
        # applying non-maximum suppresion
        detections = self._apply_nms(detections)
        logger.debug(f"Detected {len(detections)} objects total")
        return detections
    
    def _detect_object_class(self, pointcloud: np.ndarray, obj_class: str, specs: Dict) -> List[BoundingBox3D]:
        ground_z = self._estimate_ground_plane(pointcloud)
        height_min, height_max = specs['height_filter']
        height_mask = ((pointcloud[:, 2] > ground_z + height_min) & (pointcloud[:, 2] < ground_z + height_max))
        filtered_points = pointcloud[height_mask]
        
        if len(filtered_points) < specs['min_points']:
            return []
        
        # adaptive clustering on the basis of object classes
        if obj_class == 'human':
            eps = 0.3 
        else: 
            eps = 0.5
        clustering = DBSCAN(eps=eps, min_samples=specs['min_points']).fit(filtered_points[:, :3])
        labels = clustering.labels_
        
        detections = []
        unique_labels = set(labels)
        unique_labels.discard(-1) # removing noise
        
        for label in unique_labels:
            cluster_points = filtered_points[labels == label]
            if len(cluster_points) < specs['min_points']:
                continue
            
            bbox = self._compute_oriented_bbox(cluster_points, specs['size_range'])
            if bbox is not None:
                detections.append(bbox)
                
        return detections
    
    def _compute_oriented_bbox(self, points: np.ndarray, size_range: Dict) -> Optional[BoundingBox3D]:
        if len(points) < 10:
            return None
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords
        
        if not (size_range['length'][0] <= size_range['length'][1] and
                size_range['width'][0] <= size_range['width'][1] and
                size_range['height'][0] <= size_range['height'][1]
                ):
            return None
        
        # PCA for orientation estimation
        rotation = self._estimate_orientation_pca(points)
        confidence = min(1.0, len(points) / 100.0)
        
        return BoundingBox3D(
            center=center[:3],
            size=size[:3],
            rotation=rotation,
            confidence=confidence,
            timestamp=time.time()
        )
        
    def _estimate_orientation_pca(self, points: np.ndarray) -> np.ndarray:
        centered_points = points - np.mean(points, axis=0)
        xy_points = centered_points[:, :2]
        if len(xy_points) < 3:
            return np.array([0.0,0.0,0.0])
        cov_matrix = np.cov(xy_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        principal_axis = eigenvectors[:, -1] # to take the largest eigenvector need to read about it
        angle_z = np.arctan2(principal_axis[1], principal_axis[0])
        return np.array([0.0, 0.0, angle_z])
    
    def _estimate_ground_plane(self, point_cloud: np.ndarray) -> float:
        return np.percentile(point_cloud[:, 2], 5)
    
    def _apply_nms(self, detections: List[BoundingBox3D]) -> List[BoundingBox3D]:
        if len(detections) <= 1:
            return detections
        
        # trying to sort by confidence threshold
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        for i,det1 in enumerate(detections):
            overlap = False
            for j in keep:
                det2 = detections[j]
                iou = self._compute_3d_iou(det1, det2)
                if iou > self.nms_threshold:
                    overlap = True
                    break
            
            if not overlap:
                keep.append(i)
                
        return [detections[i] for i in keep]
    
    def _compute_3d_iou(self, bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> float:
        center_dist = np.linalg.norm(bbox1.center - bbox2.center)
        size_sum = np.mean(bbox1.size) + np.mean(bbox2.size)
        
        if center_dist >= size_sum:
            return 0.0
        else:
            return max(0.0, 1.0 - (center_dist / size_sum))
        
class EnhancedPointCloudTrackingSystem:
    