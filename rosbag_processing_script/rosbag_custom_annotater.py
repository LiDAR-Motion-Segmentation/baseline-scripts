#!/usr/bin/env python3

import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm

class CustomMultiSensorAnnotator:
    def __init__(self, sync_map_file, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parent=True, exist_ok=True)
        
        # new sub directories for storing the data
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "visualization").mkdir(exist_ok=True)
        (self.output_dir / "detections_2d").mkdir(exist_ok=True)
        
        with open(sync_map_file, 'r') as f:
            self.sync_map = json.load(f)
        
        # using yolov8 as of now will change in future
        self.yolo_model = YOLO('yolov8n.pt')
        
        # moving object classes , probably will have to change in the future
        self.moving_classes =  {
            0: 'person', 
            1: 'bicycle', 
            2: 'car', 
            3: 'motorcycle', 
            5: 'bus', 
            7: 'truck',
        }
        
        self.annotation = []
        
    def load_intrinsics(self, intrinsics_file):
        try:
            data = np.load(intrinsics_file)
            return {
                'D': data['D'],
                'K': data['K'],
                'height': int(data['height']),
                'width': int(data['width']),
                'fx': float(data['K'][0, 0]),
                'fy': float(data['K'][1, 1]),
                'cx': float(data['K'][0, 2]),
                'cy': float(data['K'][1, 2])
            }
        except Exception as e:
            print(f"Error in loading the intrinsics: {e}")
            return None
        
    def detect_2d_objects(self, image, confidene_threshold=0.5):
        results = self.yolo_model(image)
        detection = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf)
                    
                    if class_id in self.moving_classes and confidence >= confidene_threshold:
                        
                        # check this line once
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()
                        
                        detection.append({
                            'class_id': class_id,
                            'class_name': self.moving_classes[class_id],
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': confidence
                        })
        return detection
    
    def create_frustum_from_bbox(self, bbox, intrinsics, depth_range=(0.5, 50.0)):
        """create 3D frustum from 2D bounding box"""
        
        x1, y1, x2, y2 = bbox
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        
        near, far = depth_range
        
        # creating frustum vertices in camera coordinates
        # near plane coordinates
        corners_2d = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        frustrum_points = []
        
        # need to uunderstand this logic
        for depth in [near, far]:
            for corner in corners_2d:
                x_norm = (corner[0] - cx) / fx
                y_norm = (corner[1] - cy) / fy
                point_3d = np.array([x_norm * depth, y_norm * depth, depth])
                frustrum_points.append(point_3d)
                
        return np.array(frustrum_points)
    
    def filter_points_in_frustum(self, points, frustum_vertices):
        """to fiter points that lie in the frustum"""
        # get frustum bounding box
        min_x, max_x = frustum_vertices[:, 0].min(), frustum_vertices[:, 0].max()
        min_y, max_y = frustum_vertices[:, 1].min(), frustum_vertices[:, 1].max()
        min_z, max_z = frustum_vertices[:, 2].min(), frustum_vertices[:, 2].max()
        
        # filter points within a bounding box
        # logic might be wrong need to check
        mask = (
            (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
            (points[:, 2] >= min_z) & (points[:, 2] <= max_z) &
            (points[:, 2] > 0) # points infront of the camera        
        )
        
        return mask 
    
    def annotate_frame(self, frame_data):
        "annotating a single frame"
        frame_id = frame_data['frame_id']
        timestamp = frame_data['timestamp']
        
        pcd = o3d.io.read_point_cloud(frame_data['lidar'])
        points = np.asarray(pcd.points)
        
        if 