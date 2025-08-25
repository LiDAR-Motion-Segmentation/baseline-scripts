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
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import requests

SAM2_AVAILABLE = True
print("✅ SAM2 successfully imported")

def mkdir_parent_directory(path):
    """Create directory with parents, compatible with older Python versions"""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except TypeError:
        if not path.exists():
            os.makedirs(str(path), exist_ok=True)

def download_sam2_checkpoint(model_size='large'):
    """Download SAM2 checkpoint if not exists"""
    sam2_urls = {
        'tiny': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt',
        'small': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt',
        'base': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt',
        'large': 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt'
    }
    
    checkpoint_names = {
        'tiny': 'sam2_hiera_tiny.pt',
        'small': 'sam2_hiera_small.pt',
        'base': 'sam2_hiera_base_plus.pt',
        'large': 'sam2_hiera_large.pt'
    }
    
    checkpoint_name = checkpoint_names.get(model_size, checkpoint_names['large'])
    checkpoint_path = Path(checkpoint_name)
    
    if checkpoint_path.exists():
        print(f"✅ SAM2 checkpoint already exists: {checkpoint_name}")
        return str(checkpoint_path)
    
    print(f"📥 Downloading SAM2 {model_size} checkpoint...")
    url = sam2_urls.get(model_size, sam2_urls['large'])
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(checkpoint_path, 'wb') as f, tqdm(
            desc=f"Downloading {checkpoint_name}",
            total=total_size,
            unit='B',
            unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {checkpoint_name}")
        return str(checkpoint_path)
        
    except Exception as e:
        print(f"❌ Failed to download SAM2 checkpoint: {e}")
        return None

class FixedMultiSensorAnnotator:
    def __init__(self, sync_map_file, output_dir, use_sam2=True, sam2_model_size='large'):
        self.output_dir = Path(output_dir)
        mkdir_parent_directory(self.output_dir)
        
        # Create subdirectories
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "visualization").mkdir(exist_ok=True)
        # (self.output_dir / "visualization_camera_1").mkdir(exist_ok=True)
        # (self.output_dir / "visualization_camera_2").mkdir(exist_ok=True)
        (self.output_dir / "segmentation_masks").mkdir(exist_ok=True)
        (self.output_dir / "segmentation_masks" / "camera1").mkdir(exist_ok=True)
        (self.output_dir / "segmentation_masks" / "camera2").mkdir(exist_ok=True)
        mkdir_parent_directory(self.output_dir / "visualization_camera1")
        mkdir_parent_directory(self.output_dir / "visualization_camera2")
        
        with open(sync_map_file, 'r') as f:
            self.sync_map = json.load(f)
        
        # Initialize YOLO
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize SAM2
        self.use_sam2 = use_sam2 and SAM2_AVAILABLE
        self.sam2_predictor = None
        self.device = 'cpu'
        
        if self.use_sam2:
            try:
                self.setup_sam2(sam2_model_size)
                if self.sam2_predictor is not None:
                    print("✅ SAM2 initialized successfully")
                else:
                    print("❌ SAM2 initialization failed, using YOLO only")
                    self.use_sam2 = False
            except Exception as e:
                print(f"❌ Failed to initialize SAM2: {e}")
                self.use_sam2 = False
        
        # Moving object classes
        self.moving_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            5: 'bus', 7: 'truck', 14: 'bird', 15: 'cat', 16: 'dog'
        }
        
        self.annotation = []
        
        print(f"✅ Initialized FixedMultiSensorAnnotator")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Sync map entries: {len(self.sync_map)}")
        print(f"   YOLO model loaded: {type(self.yolo_model)}")
        print(f"   SAM2 enabled: {self.use_sam2}")
        
    def setup_sam2(self, model_size='large'):
        """Initialize SAM2 model"""
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.init()
                    self.device = 'cuda'
                    print(f"   Using CUDA: {torch.cuda.get_device_name()}")
                except Exception as cuda_error:
                    print(f"   CUDA error: {cuda_error}")
                    print("   Falling back to CPU")
                    self.device = 'cpu'
            else:
                self.device = 'cpu'
                print("   CUDA not available, using CPU")
            
            checkpoint_path = download_sam2_checkpoint(model_size)
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                print(f"❌ SAM2 checkpoint not found: {checkpoint_path}")
                return
            
            sam2_configs = {
                'tiny': 'sam2_hiera_t.yaml',
                'small': 'sam2_hiera_s.yaml', 
                'base': 'sam2_hiera_b+.yaml',
                'large': 'sam2_hiera_l.yaml'
            }
            
            config_name = sam2_configs.get(model_size, sam2_configs['large'])
            
            try:
                sam2_model = build_sam2(config_name, checkpoint_path, device=self.device)
                self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                print(f"   SAM2 model size: {model_size}")
                print(f"   SAM2 device: {self.device}")
                
            except Exception as build_error:
                print(f"❌ Failed to build SAM2 model: {build_error}")
                if self.device == 'cuda':
                    print("   Trying CPU fallback...")
                    try:
                        self.device = 'cpu'
                        sam2_model = build_sam2(config_name, checkpoint_path, device='cpu')
                        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                        print("   ✅ SAM2 initialized on CPU")
                    except Exception as cpu_error:
                        print(f"   ❌ CPU fallback also failed: {cpu_error}")
                        self.sam2_predictor = None
                else:
                    self.sam2_predictor = None
                    
        except Exception as e:
            print(f"❌ SAM2 setup failed completely: {e}")
            traceback.print_exc()
            self.sam2_predictor = None
            
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
        
    def detect_2d_objects(self, image, confidence_threshold=0.3):
        """YOLO detection with logging"""
        results = self.yolo_model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    try:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in self.moving_classes and confidence >= confidence_threshold:
                            xyxy = box.xyxy[0].cpu().numpy().flatten()
                            
                            if len(xyxy) >= 4:
                                x1, y1, x2, y2 = xyxy[:4]
                                
                                if x2 > x1 and y2 > y1:
                                    detections.append({
                                        'class_id': class_id,
                                        'class_name': self.moving_classes[class_id],
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'confidence': confidence,
                                        'mask': None,  # Will be filled by SAM2
                                        'mask_area': None,  # Will store mask area instead of mask array
                                        'mask_score': None
                                    })
                                    
                    except Exception as e:
                        print(f"Error in YOLO processing: {e}")
                        continue
        
        return detections
    
    def detect_and_segment_objects(self, image, confidence_threshold=0.3):
        """Enhanced detection with SAM2 segmentation"""
        detections = self.detect_2d_objects(image, confidence_threshold)
        
        if self.use_sam2 and self.sam2_predictor is not None and detections:
            try:
                detections = self.apply_sam2_segmentation(image, detections)
            except Exception as sam2_error:
                print(f"    ⚠️  SAM2 segmentation failed: {sam2_error}")
        
        return detections
    
    def apply_sam2_segmentation(self, image, detections):
        """Apply SAM2 segmentation to YOLO detections"""
        if self.sam2_predictor is None:
            return detections
        
        try:
            self.sam2_predictor.set_image(image)
            
            for i, detection in enumerate(detections):
                try:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = bbox
                    
                    input_box = np.array([x1, y1, x2, y2])
                    
                    masks, scores, logits = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False
                    )
                    
                    if len(masks) > 0:
                        best_mask = masks[0].astype(np.uint8)
                        
                        # Store mask for processing but remove before JSON serialization
                        detection['mask'] = best_mask
                        detection['mask_area'] = int(np.sum(best_mask))  # JSON-serializable
                        detection['mask_score'] = float(scores[0]) if len(scores) > 0 else 0.0
                        
                        print(f"    ✅ SAM2 mask generated for {detection['class_name']}: "
                              f"{detection['mask_area']} pixels, score: {detection['mask_score']:.3f}")
                        
                except Exception as mask_error:
                    print(f"    ❌ SAM2 error for detection {i}: {mask_error}")
                    continue
            
        except Exception as e:
            print(f"❌ SAM2 segmentation failed: {e}")
        
        return detections

    def create_frustum_from_bbox(self, bbox, intrinsics, depth_range=(0.5, 50.0)):
        """Create 3D frustum from 2D bounding box"""
        x1, y1, x2, y2 = bbox
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        
        near, far = depth_range
        corners_2d = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        
        frustum_points = []
        for depth in [near, far]:
            for corner in corners_2d:
                x_norm = (corner[0] - cx) / fx
                y_norm = (corner[1] - cy) / fy
                point_3d = np.array([x_norm * depth, y_norm * depth, depth])
                frustum_points.append(point_3d)
                
        return np.array(frustum_points)
    
    def filter_points_in_frustum(self, points, frustum_vertices):
        """Filter points that lie in the frustum"""
        if len(frustum_vertices) == 0:
            return np.zeros(len(points), dtype=bool)
            
        min_x, max_x = frustum_vertices[:, 0].min(), frustum_vertices[:, 0].max()
        min_y, max_y = frustum_vertices[:, 1].min(), frustum_vertices[:, 1].max()
        min_z, max_z = frustum_vertices[:, 2].min(), frustum_vertices[:, 2].max()
        
        mask = (
            (points[:, 0] >= min_x) & (points[:, 0] <= max_x) &
            (points[:, 1] >= min_y) & (points[:, 1] <= max_y) &
            (points[:, 2] >= min_z) & (points[:, 2] <= max_z) &
            (points[:, 2] > 0)
        )
        
        return mask 
    
    def annotate_frame(self, frame_data):
        """Annotating a single frame - FIXED VERSION"""
        frame_id = frame_data.get('frame_id', 'unknown')
        timestamp = frame_data.get('timestamp', 0)
        
        print(f"🎯 Annotating Frame {frame_id} (timestamp: {timestamp})")
        
        # Validate required fields
        if 'lidar' not in frame_data or frame_data['lidar'] is None:
            print(f"❌ Frame {frame_id}: Missing lidar file")
            return None
        
        lidar_file = frame_data['lidar']
        
        if not os.path.exists(lidar_file):
            print(f"❌ Frame {frame_id}: Lidar file does not exist: {lidar_file}")
            return None
        
        try:
            pcd = o3d.io.read_point_cloud(lidar_file)
            points = np.asarray(pcd.points)
            
            if len(points) == 0:
                print(f"⚠️  Frame {frame_id}: Empty point cloud")
                return None
                
            print(f"✅ Frame {frame_id}: Loaded {len(points)} points")
            
        except Exception as e:
            print(f"❌ Frame {frame_id}: Error loading point cloud: {e}")
            return None
        
        # Initialize labels (0 = static, 1 = moving)
        labels = np.zeros(len(points), dtype=np.int32)
        all_detections = []
        
        # Process Camera 1 if available
        camera1_available = (
            frame_data.get('has_camera1', False) and 
            frame_data.get('camera1_image') and 
            frame_data.get('camera1_intrinsic')
        )
        
        if camera1_available:
            camera1_image_path = frame_data.get('camera1_image')
            camera1_intrinsics_path = frame_data.get('camera1_intrinsic')
            
            print(f"  📸 Processing Camera1...")
            
            if os.path.exists(camera1_image_path) and os.path.exists(camera1_intrinsics_path):
                try:
                    img1 = cv2.imread(camera1_image_path)
                    intrinsics1 = self.load_intrinsics(camera1_intrinsics_path)
                    
                    if img1 is not None and intrinsics1 is not None:
                        print(f"     ✅ Loaded image and intrinsics")
                        
                        # Enhanced detection
                        detections1 = self.detect_and_segment_objects(img1)
                        print(f"     📊 Found {len(detections1)} moving object detections")
                        
                        for det in detections1:
                            det['camera'] = 'camera1'
                            det['image_file'] = camera1_image_path
                        
                        all_detections.extend(detections1)
                        
                        # Apply point filtering
                        for detection in detections1:
                            frustum = self.create_frustum_from_bbox(detection['bbox'], intrinsics1)
                            mask = self.filter_points_in_frustum(points, frustum)
                            points_labeled = np.sum(mask)
                            labels[mask] = 1
                            print(f"     ✅ {detection['class_name']}: labeled {points_labeled} points")
                    else:
                        print(f"     ❌ Failed to load image or intrinsics")
                        
                except Exception as e:
                    print(f"❌ Error processing camera1 for frame {frame_id}: {e}")
                    traceback.print_exc()
            else:
                print(f"     ❌ Camera1 files not found")
        else:
            print(f"  ⏭️  Camera1 not available for this frame")
        
        # Process Camera 2 if available
        camera2_available = (
            frame_data.get('has_camera2', False) and 
            frame_data.get('camera2_image') and 
            frame_data.get('camera2_intrinsic')
        )
        
        if camera2_available:
            camera2_image_path = frame_data.get('camera2_image')
            camera2_intrinsics_path = frame_data.get('camera2_intrinsic')
            
            print(f"  📸 Processing Camera2...")
            
            if os.path.exists(camera2_image_path) and os.path.exists(camera2_intrinsics_path):
                try:
                    img2 = cv2.imread(camera2_image_path)
                    intrinsics2 = self.load_intrinsics(camera2_intrinsics_path)
                    
                    if img2 is not None and intrinsics2 is not None:
                        print(f"     ✅ Loaded image and intrinsics")
                        
                        detections2 = self.detect_and_segment_objects(img2)
                        print(f"     📊 Found {len(detections2)} moving object detections")
                        
                        for det in detections2:
                            det['camera'] = 'camera2'
                            det['image_file'] = camera2_image_path
                            
                        all_detections.extend(detections2)
                        
                        for detection in detections2:
                            frustum = self.create_frustum_from_bbox(detection['bbox'], intrinsics2)
                            mask = self.filter_points_in_frustum(points, frustum)
                            points_labeled = np.sum(mask)
                            labels[mask] = 1
                            print(f"     ✅ {detection['class_name']}: labeled {points_labeled} points")
                    else:
                        print(f"     ❌ Failed to load image or intrinsics")
                        
                except Exception as e:
                    print(f"❌ Error processing camera2 for frame {frame_id}: {e}")
                    traceback.print_exc()
            else:
                print(f"     ❌ Camera2 files not found")
        else:
            print(f"  ⏭️  Camera2 not available for this frame")
        
        print(f"🏁 Frame {frame_id} Summary:")
        print(f"   Total detections: {len(all_detections)}")
        print(f"   Moving points labeled: {np.sum(labels == 1)}")
        print(f"   Static points: {np.sum(labels == 0)}")
        
        # Save labels
        label_file = self.output_dir / "labels" / f"{timestamp:019d}.label"
        labels.astype(np.uint32).tofile(str(label_file))
        print(f"  💾 Saved labels: {label_file}")
        
        # Create visualizations BEFORE removing masks
        self.create_visualization(points, labels, frame_data, all_detections, timestamp)
        
        # Remove numpy arrays before JSON serialization
        json_safe_detections = []
        for det in all_detections:
            json_det = det.copy()
            # Remove the numpy mask but keep mask_area and mask_score
            if 'mask' in json_det:
                del json_det['mask']
            json_safe_detections.append(json_det)
        
        # Create annotation data
        annotation_data = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'lidar_file': frame_data['lidar'],
            'camera1_image': frame_data.get('camera1_image'),
            'camera2_image': frame_data.get('camera2_image'),
            'detections_2d': json_safe_detections,  # JSON-safe version
            'num_points': len(points),
            'moving_points': int(np.sum(labels == 1)),
            'static_points': int(np.sum(labels == 0)),
            'moving_ratio': float(np.sum(labels == 1) / len(points)) if len(points) > 0 else 0.0,
            'sam2_enabled': self.use_sam2,
            'sam2_working': self.sam2_predictor is not None
        }
        
        return annotation_data
    
    def create_visualization(self, points, labels, frame_data, detections, timestamp):
        """Create all visualizations including SAM2 masks"""
        
        # 1. Save labeled point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        colors = np.zeros((len(points), 3))
        colors[labels == 0] = [0.7, 0.7, 0.7]  # Static - gray
        colors[labels == 1] = [1.0, 0.0, 0.0]  # Moving - red
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        viz_file = self.output_dir / "visualization" / f"{timestamp:019d}_labeled.pcd"
        success = o3d.io.write_point_cloud(str(viz_file), pcd)
        if success:
            print(f"  💾 Saved point cloud visualization: {viz_file.name}")
        
        # 2. Create camera visualizations with detections
        for camera_name in ['camera1', 'camera2']:
            self.create_camera_visualization(frame_data, detections, timestamp, camera_name)
        
        # 3. Save SAM2 masks separately
        if self.use_sam2:
            self.save_sam2_masks(detections, timestamp)
    
    def create_camera_visualization(self, frame_data, detections, timestamp, camera_name):
        """Create visualization for a single camera"""
        image_key = f'{camera_name}_image'
        if not (frame_data.get(image_key) and os.path.exists(frame_data[image_key])):
            return
        
        try:
            img = cv2.imread(frame_data[image_key])
            if img is None:
                return
            
            camera_detections = [d for d in detections if d.get('camera') == camera_name]
            
            # Color mapping for different classes
            color_map = {
                'person': (0, 255, 0),        # Green
                'bicycle': (0, 255, 255),     # Yellow
                'car': (255, 0, 0),           # Blue
                'motorcycle': (255, 0, 255), # Magenta
                'bus': (0, 165, 255),         # Orange
                'truck': (128, 0, 128),       # Purple
            }
            
            # Create mask overlay
            mask_overlay = np.zeros_like(img, dtype=np.uint8)
            
            # Draw detections
            for det in camera_detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                class_name = det['class_name']
                confidence = det['confidence']
                color = color_map.get(class_name, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Draw SAM2 mask overlay if available
                if self.use_sam2 and det.get('mask') is not None:
                    mask = det['mask']
                    colored_mask = np.zeros_like(img, dtype=np.uint8)
                    colored_mask[mask > 0] = color
                    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.4, 0)
                
                # Draw label
                label_parts = [f"{class_name}: {confidence:.2f}"]
                if det.get('mask_score') is not None:
                    label_parts.append(f"mask: {det['mask_score']:.2f}")
                label = " | ".join(label_parts)
                
                # Label background and text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                
                cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
                cv2.putText(img, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
            
            # Blend mask overlay with image
            if self.use_sam2:
                img = cv2.addWeighted(img, 0.7, mask_overlay, 0.3, 0)
            
            # Add info text
            info_parts = [f"Frame: {timestamp}", f"{camera_name.upper()}", f"Detections: {len(camera_detections)}"]
            if self.use_sam2:
                masks_count = sum(1 for d in camera_detections if d.get('mask') is not None)
                info_parts.append(f"SAM2 Masks: {masks_count}")
            
            info_text = " | ".join(info_parts)
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            
            # Add border if detections found
            if camera_detections:
                border_color = (0, 255, 255) if self.use_sam2 else (0, 255, 0)
                cv2.rectangle(img, (0, 0), (img.shape[1]-1, img.shape[0]-1), border_color, 5)
            
            # Save visualization
            output_dir = self.output_dir / f"visualization_{camera_name}"
            mkdir_parent_directory(output_dir)
            viz_file = output_dir / f"{timestamp:019d}.png"
            success = cv2.imwrite(str(viz_file), img)
            
            if success:
                print(f"  💾 Saved {camera_name} visualization: {viz_file.name}")
            else:
                print(f"  ❌ Failed to save {camera_name} visualization: {viz_file.name}")
            
        except Exception as e:
            print(f"  ❌ Error creating {camera_name} visualization: {e}")
    
    def save_sam2_masks(self, detections, timestamp):
        """Save SAM2 segmentation masks for each camera separately"""
        try:
            for camera_name in ['camera1', 'camera2']:
                camera_detections = [d for d in detections if d.get('camera') == camera_name]
                detections_with_masks = [d for d in camera_detections if d.get('mask') is not None]
                
                if detections_with_masks:
                    print(f"  💾 Saving SAM2 masks for {camera_name}: {len(detections_with_masks)} masks")
                    
                    # Get image dimensions from first mask
                    first_det = detections_with_masks[0]
                    h, w = first_det['mask'].shape
                    
                    # Create combined mask image
                    combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                             (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)]
                    
                    # Save individual masks and create combined mask
                    for i, det in enumerate(detections_with_masks):
                        if det.get('mask') is not None:
                            mask = det['mask']
                            color = colors[i % len(colors)]
                            class_name = det['class_name']
                            confidence = det['confidence']
                            
                            # Save individual mask
                            individual_mask = mask * 255  # Convert to 0-255 range
                            mask_dir = self.output_dir / "segmentation_masks" / camera_name
                            individual_file = mask_dir / f"{timestamp:019d}_{i:02d}_{class_name}_{confidence:.2f}.png"
                            cv2.imwrite(str(individual_file), individual_mask)
                            
                            # Add to combined mask with unique color
                            for c in range(3):
                                combined_mask[:, :, c][mask > 0] = color[c]
                    
                    # Save combined mask
                    combined_file = self.output_dir / "segmentation_masks" / f"{timestamp:019d}_{camera_name}_combined.png"
                    cv2.imwrite(str(combined_file), combined_mask)
                    
                    print(f"    ✅ Saved individual and combined masks for {camera_name}")
                        
        except Exception as e:
            print(f"❌ Error saving SAM2 masks: {e}")
            traceback.print_exc()
                
    def process_all_frames(self, max_frames=None):
        """Process frames with proper error handling"""
        frames_to_process = self.sync_map[:max_frames] if max_frames else self.sync_map
        
        print(f"🚀 Processing {len(frames_to_process)} frames...")
        print(f"   SAM2 enabled: {self.use_sam2}")
        
        for frame_data in tqdm(frames_to_process, desc="Annotating frames"):
            try:
                annotation_data = self.annotate_frame(frame_data)
                if annotation_data:
                    self.annotation.append(annotation_data)
            except Exception as e:
                print(f"❌ Error processing frame {frame_data.get('frame_id', 'unknown')}: {e}")
                traceback.print_exc()
                continue
        
        self.save_annotation_summary()
        self.print_statistics()
        
    def save_annotation_summary(self):
        """Save annotation summary - JSON safe version"""
        summary_file = self.output_dir / "annotation_summary.json"
        
        try:
            with open(str(summary_file), 'w') as f:
                json.dump(self.annotation, f, indent=2)
            print(f"📄 Annotation summary saved to: {summary_file}")
        except Exception as e:
            print(f"❌ Error saving JSON summary: {e}")
            # Try to save a simplified version
            try:
                simplified_annotations = []
                for ann in self.annotation:
                    simplified_ann = {
                        'frame_id': ann['frame_id'],
                        'timestamp': ann['timestamp'],
                        'num_points': ann['num_points'],
                        'moving_points': ann['moving_points'],
                        'static_points': ann['static_points'],
                        'moving_ratio': ann['moving_ratio'],
                        'num_detections': len(ann['detections_2d']),
                        'sam2_enabled': ann['sam2_enabled']
                    }
                    simplified_annotations.append(simplified_ann)
                
                simplified_file = self.output_dir / "annotation_summary_simplified.json"
                with open(str(simplified_file), 'w') as f:
                    json.dump(simplified_annotations, f, indent=2)
                print(f"📄 Simplified annotation summary saved to: {simplified_file}")
                
            except Exception as e2:
                print(f"❌ Failed to save even simplified summary: {e2}")
        
    def print_statistics(self):
        """Print comprehensive statistics"""
        if not self.annotation:
            print("No annotations generated")
            return
        
        total_points = sum(ann['num_points'] for ann in self.annotation)
        total_moving = sum(ann['moving_points'] for ann in self.annotation)
        total_static = sum(ann['static_points'] for ann in self.annotation)
        total_detections = sum(len(ann['detections_2d']) for ann in self.annotation)
        frames_with_detections = sum(1 for ann in self.annotation if ann['detections_2d'])
        
        print(f"\n📊 Annotation Statistics:")
        print(f"   Total frames processed: {len(self.annotation)}")
        print(f"   Frames with detections: {frames_with_detections}")
        print(f"   Total 2D detections: {total_detections}")
        print(f"   Total points: {total_points:,}")
        print(f"   Moving points: {total_moving:,} ({total_moving/total_points*100:.1f}%)")
        print(f"   Static points: {total_static:,} ({total_static/total_points*100:.1f}%)")
        
        if self.annotation:
            avg_moving_ratio = np.mean([ann['moving_ratio'] for ann in self.annotation])
            print(f"   Average moving ratio: {avg_moving_ratio*100:.1f}%")
            
            sam2_working = sum(1 for ann in self.annotation if ann.get('sam2_working', False))
            print(f"   SAM2 working frames: {sam2_working}/{len(self.annotation)}")
            
            if self.use_sam2:
                total_masks = 0
                for ann in self.annotation:
                    for det in ann['detections_2d']:
                        if det.get('mask_area', 0) > 0:
                            total_masks += 1
                print(f"   Total SAM2 masks generated: {total_masks}")

def main():
    parser = argparse.ArgumentParser(description="Fixed multi-sensor annotator")
    parser.add_argument("sync_map", help="Path to synchronization map JSON file")
    parser.add_argument("--output", "-o", default="fixed_annotations", help="Output directory")
    parser.add_argument("--max_frames", "-n", type=int, help="Maximum frames to process")
    parser.add_argument("--disable_sam2", action="store_true", help="Disable SAM2 segmentation")
    parser.add_argument("--sam2_model", choices=['tiny', 'small', 'base', 'large'], 
                        default='small', help="SAM2 model size")
    
    args = parser.parse_args()
    
    annotator = FixedMultiSensorAnnotator(
        args.sync_map, 
        args.output, 
        use_sam2=not args.disable_sam2,
        sam2_model_size=args.sam2_model
    )
    annotator.process_all_frames(max_frames=args.max_frames)

if __name__ == "__main__":
    main()
