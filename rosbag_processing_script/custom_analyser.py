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
                         lidar_files) -> list:
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
    
    def load_instrinsics(self, intrinsic_file: Path):
        try:
            data = np.load(intrinsic_file)
            intrinsic = {
                'D': data['D'] if 'D' in data else np.zeros(5),
                'K': data['K'] if 'K' in data else np.eye(3),
                'height': int(data['height']),
                'width': int(data['width'])
            }
            K = np.array(intrinsic['K'])
            intrinsic['fx'] = float(K[0, 0])
            intrinsic['fy'] = float(K[1, 1])
            intrinsic['cx'] = float(K[0, 2])
            intrinsic['cy'] = float(K[1, 2])            
            return intrinsic
        
        except Exception as e:
            print (f"error loading intrinsic file {intrinsic_file}: {e}")
            return None
        
    def save_sync_data(self, sync_data, output_file):
        """saving the synced data to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(sync_data, f, indent=2)
        print(f"syncronised map saved to : {output_file}")
              
def main():
    parser = argparse.ArgumentParser(description="Analyse custom bag data structure")
    parser.add_argument("bag_path", help="Path to the bag directory")
    parser.add_argument("--output", "-o", help="Output JSON file for sync map")
    args =  parser.parse_args()
    
    analyzer = CustomDataAnalyzer(args.bag_path)
    sync_data = analyzer.analyse_data()
    
    if sync_data:
        output_file = args.output if args.output else f"{analyzer.bag_name}_sync_map.json"
        analyzer.save_sync_data(sync_data, output_file)
        
        # printing a sample to check
        try:
            sample_entry = sync_data[0] 
        except:
            print(f"the sample entry doesnt exist {sample_entry}")
            return None
        
        if sample_entry and sample_entry.get('camera1_intrinsics'):
            print("\n Sample Camera1 Intrinsics:")
            intrinsics = analyzer.load_intrinsics(sample_entry['camera1_intrinsics'])
            if intrinsics:
                print(f"   Resolution: {intrinsics['width']}x{intrinsics['height']}")
                print(f"   Focal length: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
                print(f"   Principal point: cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
                
        return output_file
    return None

if __name__ == "__main__":
    main()
    print(f"The code starts")