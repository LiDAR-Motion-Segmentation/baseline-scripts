#!/usr/bin/env python3

import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
import json
import argparse
from ultralytics import YOLO
from tqdm import tqdm
import os
import traceback

def mkdir_parent_directory(path):
    """Create directory with parents, compatible with older Python versions"""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except TypeError:
        # Fallback for older pathlib versions
        if not path.exists():
            os.makedirs(str(path), exist_ok=True)

class CustomMultiSensorAnnotator:
    def __init__(self, sync_map_file, output_dir):
        self.output_dir = Path(output_dir)
        mkdir_parent_directory(self.output_dir)
        
        # new sub directories for storing the data
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "visualization").mkdir(exist_ok=True)
        (self.output_dir / "visualization_camera_1").mkdir(exist_ok=True)
        (self.output_dir / "visualization_camera_2").mkdir(exist_ok=True)
        
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
            14: 'bird',
            15: 'cat',
            16: 'dog'
        }
        
        self.annotation = []
        
        print(f" Initialized CustomMultiSensorAnnotator")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Sync map entries: {len(self.sync_map)}")
        print(f"   YOLO model loaded: {type(self.yolo_model)}")
        
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
                    try:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf)
                        
                        if class_id in self.moving_classes and confidence >= confidene_threshold:
                            
                            # check this line once
                            # This gives [x1, y1, x2, y2]
                            # x1, y1, x2, y2 = box.xyxy.cpu().numpy()
                            xyxy = box.xyxy.cpu().numpy()
                            # print(f" YOLO bounding box check: {xyxy}, shape: {xyxy.shape}")
                            
                            # if len(xyxy) == 4:
                            x1, y1, x2, y2 = xyxy.flatten()
                            
                            if x2 > x1 and y2 > y1:
                                detection.append({
                                    'class_id': class_id,
                                    'class_name': self.moving_classes[class_id],
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': confidence
                                })
                            else:
                                print(f"Invalid bbox coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                                
                    except Exception as e:
                        print(f"error in processing yolo code: {e}")
                        continue
                    
        return detection
    
    def create_frustum_from_bbox(self, bbox, intrinsics, depth_range=(0.5, 50.0)):
        """create 3D frustum from 2D bounding box"""
        
        # if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        #     x1, y1, x2, y2 = bbox
        # elif hasattr(bbox, 'shape') and bbox.shape == (4,):
        #     # If it's a numpy array
        x1, y1, x2, y2 = bbox
        # else:
        #     print(f" Invalid bbox format: {bbox}")
        #     return np.array([]) 
            
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
        frame_id = frame_data.get('frame_id', 'unknown')
        timestamp = frame_data.get('timestamp', 0)
        
        # Validate required fields
        if 'lidar' not in frame_data or frame_data['lidar'] is None:
            print(f" Frame {frame_id}: Missing lidar file")
            return None
        
        lidar_file = frame_data['lidar']
        
        if not os.path.exists(lidar_file):
            print(f" Frame {frame_id}: Lidar file does not exist: {lidar_file}")
            return None
        
        try:
            pcd = o3d.io.read_point_cloud(frame_data['lidar'])
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                print(f"Warning: Empty point cloud for frame {frame_id}")
                return None
        
        except Exception as e:
            print(f" Frame {frame_id}: Error loading point cloud: {e}")
            return None
            
        # labels intitialization (0 = static, 1 = moving)
        labels = np.zeros(len(points), dtype=np.int32)
        
        all_detection = []
        
        # process camera1 if available 
        if frame_data['has_camera1'] and frame_data['camera1_intrinsic']:
            try:
                img1 = cv2.imread(frame_data['camera1_image'])
                intrinsics1 = self.load_intrinsics(frame_data['camera1_intrinsic'])
                if img1 is not None and intrinsics1 is not None:
                    # print(f"Loaded intrinsics: fx={intrinsics1['fx']:.1f}, fy={intrinsics1['fy']:.1f}")
                    detection1 = self.detect_2d_objects(img1)
                    print(f" Found {len(detection1)} detections")
                    
                    for det in detection1:
                        det['camera'] = 'camera1'
                        det['image_file'] = frame_data['camera1_image']
                    
                    all_detection.extend(detection1)
                    
                    # apply frustum-based labelling
                    for detection in detection1:
                        frustum = self.create_frustum_from_bbox(detection['bbox'], intrinsics1)
                        mask = self.filter_points_in_frustum(points, frustum)
                        labels[mask] = 1 # marking moving obstacles
                        
            except Exception as e:
                print(f"Error processing camera1 for frame {frame_id}: {e}")
        
        # process camera2 if available 
        if frame_data['has_camera2'] and frame_data['camera2_intrinsic']:
            try:
                img2 = cv2.imread(frame_data['camera2_image'])
                intrinsics2 = self.load_intrinsics(frame_data['camera2_intrinsic'])
                if img2 is not None and intrinsics2 is not None:
                    detection2 = self.detect_2d_objects(img2)
                    
                    for det in detection2:
                        det['camera'] = 'camera2'
                        det['image_file'] = frame_data['camera2_image']
                        
                    all_detection.extend(detection2)
                    
                    # apply frustum-based labelling
                    for detection in detection2:
                        frustum = self.create_frustum_from_bbox(detection['bbox'], intrinsics2)
                        mask = self.filter_points_in_frustum(points, frustum)
                        labels[mask] = 1 # marking moving obstacles
                        
            except Exception as e:
                print(f"Error processing camera2 for frame {frame_id}: {e}")
                
        # create annotation data
        annotation_data = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'lidar_file': frame_data['lidar'],
            'camera1_image': frame_data.get('camera1_image'),
            'camera2_image': frame_data.get('camera2_image'),
            'detections_2d': all_detection,
            'num_points': len(points),
            'moving_points': int(np.sum(labels == 1)),
            'static_points': int(np.sum(labels == 0)),
            'moving_ratio': float(np.sum(labels == 1) / len(points)) if len(points) > 0 else 0.0
        }
        
        # Save labels in binary format (compatible with SemanticKITTI)
        label_file = self.output_dir / "labels" / f"{timestamp:019d}.label"
        labels.astype(np.uint32).tofile(label_file)
        
        self.create_visualization(points, labels, frame_data, all_detection)
        return annotation_data
    
    def create_visualization(self, points, labels, frame_data, detections):
        frame_id = frame_data['frame_id']
        timestamp = frame_data['timestamp']
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # static grey colour , moving as red colour 
        colours = np.zeros((len(points), 3))
        colours[labels == 0] = [0.7,0.7,0.7]
        colours[labels == 1] = [1.0,0.0,0.0]
        
        # need to understand this portion
        pcd.colors = o3d.utility.Vector3dVector(colours)
        
        viz_file = self.output_dir / "visualization" / f"{timestamp:019d}_labeled.pcd"
        saved_pcd = o3d.io.write_point_cloud(str(viz_file), pcd)
        if saved_pcd:
            print(f"Saved visualization file {saved_pcd}")
        else:
            print("failed to save visualization file")
            
        self.create_camera_visualization(frame_data, detections, timestamp)
        
    def create_camera_visualization(self, frame_data, detections, timestamp):
        camera1_viz_dir = self.output_dir / "visualization_camera_1"
        camera2_viz_dir = self.output_dir / "visualization_camera_2"
        
        mkdir_parent_directory(camera1_viz_dir)
        mkdir_parent_directory(camera1_viz_dir)
        
        # process Camera 1
        if frame_data.get('has_camera1') and frame_data.get('camera1_image'):
            camera1_image = frame_data['camera1_image']
            if os.path.exists(camera1_image):
                camera1_detections = [d for d in detections if d.get('camera') == 'camera1']
                self.process_single_camera_visualization(
                    camera_name='camera1',
                    image_file=frame_data['camera1_image'],
                    detections=detections,
                    timestamp=timestamp,
                    output_dir=camera1_viz_dir
            )
            else:
                print(f" Camera1 image not found: {camera1_image}")
        
        # Process Camera 2
        if frame_data.get('has_camera2') and frame_data.get('camera2_image'):
            camera2_image = frame_data['camera2_image']
            if os.path.exists(camera2_image):
                camera2_detections = [d for d in detections if d.get('camera') == 'camera2']
                self.process_single_camera_visualization(
                    camera_name='camera2',
                    image_file=frame_data['camera2_image'],
                    detections=detections,
                    timestamp=timestamp,
                    output_dir=camera2_viz_dir
                )
            else:
                print(f" Camera2 image not found: {camera2_image}")
            
    def process_single_camera_visualization(self, camera_name, image_file, detections, timestamp, output_dir):
        try:
            # Check if image file exists
            if not os.path.exists(image_file):
                print(f" Image file not found: {os.path.basename(image_file)}")
                return
            
            # Load image
            img = cv2.imread(image_file)
            if img is None:
                print(f" Failed to load image: {os.path.basename(image_file)}")
                return
            
            camera_detections = [d for d in detections if d.get('camera') == camera_name]
            
            print(f" Processing {camera_name}: {len(camera_detections)} detections")
            # Draw detections
            for i, det in enumerate(camera_detections):
                try:
                    bbox = det['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    confidence = det['confidence']
                    class_name = det['class_name']
                    
                    # Color mapping for different classes
                    color_map = {
                        'person': (0, 255, 0),        # Green
                        'car': (255, 0, 0),           # Blue
                        'bicycle': (0, 255, 255),     # Yellow
                        'motorcycle': (255, 0, 255), # Magenta
                        'bus': (0, 165, 255),         # Orange
                        'truck': (128, 0, 128),       # Purple
                        'bird': (255, 255, 0),        # Cyan
                        'cat': (255, 192, 203),       # Pink
                        'dog': (165, 42, 42)          # Brown
                    }
                    
                    color = color_map.get(class_name, (0, 255, 0))  # Default green
                    
                    # Draw bounding box with thicker lines
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    
                    # Prepare label text
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Get text size for background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                    
                    # Draw label background
                    label_bg_start = (x1, y1 - text_height - 10)
                    label_bg_end = (x1 + text_width + 10, y1)
                    cv2.rectangle(img, label_bg_start, label_bg_end, color, -1)
                    
                    # Draw label text
                    cv2.putText(img, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
                    
                    print(f"Drew detection {i+1}: {class_name} at [{x1}, {y1}, {x2}, {y2}] conf:{confidence:.2f}")
                    
                except Exception as draw_error:
                    print(f" Error drawing detection {i+1}: {draw_error}")
                    continue
            
            # Add frame information overlay
            info_text = f"Frame: {timestamp} | Camera: {camera_name.upper()} | Detections: {len(camera_detections)}"
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add border around image if there are detections
            if camera_detections:
                cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 255, 0), 5)
            
            # Save visualization as .png
            viz_filename = f"{timestamp:019d}_{camera_name}.png"
            viz_file = output_dir / viz_filename
            
            success = cv2.imwrite(str(viz_file), img)
            
            if success:
                print(f" Saved {camera_name} visualization: {viz_filename}")
            else:
                print(f"Failed to save {camera_name} visualization: {viz_filename}")
                
        except Exception as camera_error:
            print(f" Error processing {camera_name} visualization: {camera_error}")
            traceback.print_exc()
                
    def process_all_frames(self, max_frames=None):
        # if max_frames:
        #     frames_to_process = self.sync_map[:max_frames]
        # else:
        #     frames_to_process = self.sync_map
        frames_to_process = self.sync_map[:max_frames] if max_frames else self.sync_map
            
        print(f"Processing {len(frames_to_process)} frames...")
        
        for frame_data in tqdm(frames_to_process, desc="Annotating frames"):
            try:
                annotation_data = self.annotate_frame(frame_data)
                if annotation_data:
                    self.annotation.append(annotation_data)
            except Exception as e:
                print(f" Error processing frame {frame_data['frame_id']}: {e}")
                continue
            
        self.save_annotation_summary()
        self.print_statistics()
        
    def save_annotation_summary(self):
        summary_file = self.output_dir / "annotation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.annotation, f, indent=2)
            
        print(f"Annotation summary saved to: {summary_file}")
        
    def print_statistics(self):
        if not self.annotation:
            print("No annotations generated")
            return
        
        total_points = sum(ann['num_points'] for ann in self.annotation)
        total_moving = sum(ann['moving_points'] for ann in self.annotation)
        total_static = sum(ann['static_points'] for ann in self.annotation)
        
        total_detections = sum(len(ann['detections_2d']) for ann in self.annotation)
        frames_with_detections = sum(1 for ann in self.annotation if ann['detections_2d'])
        
        print(f"\n Annotation Statistics:")
        print(f"   Total frames processed: {len(self.annotation)}")
        print(f"   Frames with detections: {frames_with_detections}")
        print(f"   Total 2D detections: {total_detections}")
        print(f"   Total points: {total_points:,}")
        print(f"   Moving points: {total_moving:,} ({total_moving/total_points*100:.1f}%)")
        print(f"   Static points: {total_static:,} ({total_static/total_points*100:.1f}%)")
        
        # average moving ratio
        avg_moving_ratio = np.mean([ann['moving_ratio'] for ann in self.annotation])
        print(f"   Average moving ratio: {avg_moving_ratio*100:.1f}%")
        
def main():
    parser = argparse.ArgumentParser(description= "Custom multi-sensor annotator")
    parser.add_argument("sync_map", help="Path to synchronization map JSON file")
    parser.add_argument("--output", "-o", default="annotations", help="Output directory")
    parser.add_argument("--max_frames", "-n", type=int, help="Maximum frames to process")
    args = parser.parse_args()
    annotator = CustomMultiSensorAnnotator(args.sync_map, args.output)
    annotator.process_all_frames(max_frames=args.max_frames)
        
if __name__ == "__main__":
    main()