#!/usr/bin/env python3
import os
import numpy as np
import cv2
from pathlib import Path
import json
from collections import defaultdict
import argparse

class CustomDataAnalyzer:
    def __init__(self, bag_path):
        self.bag_path = Path(bag_path)
        self.bag_name = self.bag_path.name
        
        self.camera1_images_path = self.bag_path / "camera1_images"
        self.camera2_images_path = self.bag_path / "camera2_images"
        self.camera1_intrinsics_path = self.bag_path / "camera1_intrinsics"
        self.camera2_intrinsics_path = self.bag_path / "camera2_intrinsics"
        self.lidar_path = self.bag_path / "lidar"
        
    def validate_structure(self) -> bool:
        requires_dirs = [
            self.camera1_images_path,
            self.camera2_images_path,
            self.camera1_intrinsics_path,
            self.camera2_intrinsics_path,
            self.lidar_path
        ]
        
        misssing_dirs = []
        for dir_path in requires_dirs:
            if not dir_path.exists():
                misssing_dirs.append(dir_path)
                
        if misssing_dirs:
            print(f"missing directories in {misssing_dirs}")
            return False
        
        print(f" all the required directories were found in {self.bag_name}")
        return True
    
    def analyse_data(self):
        """Analyze all data files and extract timestamps"""
        if not self.validate_structure():
            return None
    
        camera1_imgs = sorted(self.camera1_images_path.glob("*.png"))
        camera2_imgs = sorted(self.camera2_images_path.glob("*.png"))
        camera1_intrinsics = sorted(self.camera1_intrinsics_path.glob("*.npz"))
        camera2_intrinsics = sorted(self.camera2_intrinsics_path.glob("*.npz"))
        lidar_files = sorted(self.lidar_path.glob("*.pcd"))
        
        print(f" Data Analysis for {self.bag_name}:")
        print(f"   Camera1 images: {len(camera1_imgs)}")
        print(f"   Camera1 intrinsics: {len(camera1_intrinsics)}")
        print(f"   Camera2 images: {len(camera2_imgs)}")
        print(f"   Camera2 intrinsics: {len(camera2_intrinsics)}")
        print(f"   LiDAR point clouds: {len(lidar_files)}")
        
        sync_data = self.create_sync_data(camera1_imgs, camera1_intrinsics,
                                          camera2_imgs, camera2_intrinsics, lidar_files)
        
        return sync_data
    
    def extract_timestamps(self, file_path: Path) -> int:
        """extract timestamp from filename"""
        try:
            timestamp = int(file_path.stem)
            return timestamp
        except ValueError:
            print(f"Inavalid timestamp in file: {file_path}")
            return 0
        
    def create_sync_data(self, 
                         cam1_imgs,
                         cam1_intrinsics,
                         cam2_imgs,
                         cam2_intrinsics,
                         lidar_files):
        """create a sycchronized data"""
        timestamp_data = defaultdict(dict) #to group data by timestamp
        
        for img_file in cam1_imgs:
            timestamp = self.extract_timestamps(img_file)
            timestamp_data[timestamp]['camera1_image'] = str(img_file)
            
        for intrinsic_file in cam1_intrinsics:
            timestamp = self.extract_timestamps(intrinsic_file)
            timestamp_data[timestamp]['camera1_intrinsic'] = str(intrinsic_file)
            
        for img_file in cam2_imgs:
            timestamp = self.extract_timestamps(img_file)
            timestamp_data[timestamp]['camera2_image'] = str(img_file)
            
        for intrinsic_file in cam2_intrinsics:
            timestamp = self.extract_timestamps(intrinsic_file)
            timestamp_data[timestamp]['camera2_intrinsic'] = str(intrinsic_file)
            
        for lidar_file in lidar_files:
            timestamp = self.extract_timestamps(lidar_file)
            timestamp_data[timestamp]['lidar'] = str(lidar_file)
            
        sync_data = []
        for timestamp in sorted(timestamp_data.keys()):
            data = timestamp_data[timestamp]
            if 'lidar' in data:
                sync_entry = {
                    'frame_id': len(sync_data),
                    'timestamp': timestamp,
                    'camera1_image': data.get('camera1_image'),
                    'camera1_instrinsic': data.get('camera1_intrinsic'),
                    'camera2_image': data.get('camera2_image'),
                    'camera2_instrinsic': data.get('camera2_intrinsic'),
                    'has_camera1': 'camera1_image' in data,
                    'has_camera2': 'camera2_image' in data,
                    'complete_frame': all(k in data for k in ['lidar', 'camera1_image', 'camera2_image'])
                }
                sync_data.append(sync_entry)
            
        print(f" Synchronized {len(sync_data)} frames")
        complete_frames = sum(1 for entry in sync_data if entry['complete_frame'])
        print(f" Complete frames (LiDAR + both cameras): {complete_frames}")
        return sync_data
    
    def 