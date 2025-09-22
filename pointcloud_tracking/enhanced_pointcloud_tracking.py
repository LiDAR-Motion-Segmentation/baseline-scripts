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
        self.object_charactertics = {}
        
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
                'avg_size': bbox
            }