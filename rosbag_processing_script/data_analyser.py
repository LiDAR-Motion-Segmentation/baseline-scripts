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
    
    