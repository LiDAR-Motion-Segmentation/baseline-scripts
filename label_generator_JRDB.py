#!/usr/bin/env python3
"""
Convert single JRDB 3D JSON file (containing sequence data) to multiple KITTI MOS format label files
Processes one JSON file with multiple point clouds and generates individual .label files matching velodyne naming
python3 label_generator_JRDB.py --json_file /scratch/soumo_roy/JRDB/labels/labels_3d/bytes-cafe-2019-02-07_0.json --bin_folder /scratch/soumo_roy/JRDB/kitty-format/sequences/00/velodyne --output_folder /scratch/soumo_roy/JRDB/kitty-format/sequences/00/labels --verify
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import struct

def load_point_cloud(bin_file):
    """
    Load point cloud from .bin file in KITTI format
    Returns: numpy array of shape (N, 4) where each point is [x, y, z, intensity]
    """
    try:
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        return points
    except Exception as e:
        print(f"Error loading point cloud {bin_file}: {e}")
        return None

def point_in_3d_box(points, box_center, box_dimensions, rotation_z):
    """
    Check if points are inside a 3D bounding box
    
    Args:
        points: Nx3 array of point coordinates
        box_center: [cx, cy, cz] - center of the bounding box
        box_dimensions: [l, w, h] - length, width, height
        rotation_z: rotation around z-axis in radians
    
    Returns:
        Boolean array indicating which points are inside the box
    """
    # Transform points to box coordinate system
    cx, cy, cz = box_center
    l, w, h = box_dimensions
    
    # Translate points to box center
    translated_points = points[:, :3] - np.array([cx, cy, cz])
    
    # Rotate points by -rotation_z to align with box axes
    cos_r = np.cos(-rotation_z)
    sin_r = np.sin(-rotation_z)
    rotation_matrix = np.array([
        [cos_r, -sin_r, 0],
        [sin_r, cos_r, 0],
        [0, 0, 1]
    ])
    
    rotated_points = translated_points @ rotation_matrix.T
    
    # Check if points are within box bounds
    inside_x = np.abs(rotated_points[:, 0]) <= l / 2
    inside_y = np.abs(rotated_points[:, 1]) <= w / 2
    inside_z = np.abs(rotated_points[:, 2]) <= h / 2
    
    return inside_x & inside_y & inside_z

def determine_motion_state(action_labels, social_activity):
    """
    Determine if an object is moving based on JRDB action and social activity labels
    
    Args:
        action_labels: dict of action labels with confidence scores
        social_activity: dict of social activity labels with confidence scores
    
    Returns:
        bool: True if object is considered moving, False if static
    """
    # Define motion-indicating activities (with threshold for confidence)
    motion_activities = {
        'walking': 0.5,
        'running': 0.3,
        'cycling': 0.3,
        'moving': 0.5,
        'dancing': 0.5,
        'playing': 0.4
    }
    
    # Define static activities
    static_activities = {
        'sitting': 0.7,
        'standing': 0.6,
        'lying': 0.8,
        'sleeping': 0.8
    }
    
    # Check action labels for motion indicators
    if action_labels:
        for action, confidence in action_labels.items():
            action_lower = action.lower()
            for motion_key, threshold in motion_activities.items():
                if motion_key in action_lower and confidence >= threshold:
                    return True
    
    # Check social activity for motion indicators
    if social_activity:
        for activity, confidence in social_activity.items():
            activity_lower = activity.lower()
            
            # Check for motion activities
            for motion_key, threshold in motion_activities.items():
                if motion_key in activity_lower and confidence >= threshold:
                    return True
            
            # Check for static activities (higher confidence = more likely static)
            for static_key, threshold in static_activities.items():
                if static_key in activity_lower and confidence >= threshold:
                    return False
    
    # Default: if walking confidence is low or sitting confidence is high, consider static
    # This handles the common case in JRDB where pedestrians might be stationary
    if social_activity:
        walking_conf = social_activity.get('walking', 0.0)
        sitting_conf = social_activity.get('sitting', 0.0)
        
        if sitting_conf > walking_conf and sitting_conf > 1.2:
            return False  # Likely sitting/stationary
        elif walking_conf > 1.5:
            return True   # Likely walking/moving
    
    # Conservative default: assume static unless clear evidence of motion
    return False

def get_bin_filename_from_pcd(pcd_filename):
    """
    Convert PCD filename to corresponding BIN filename
    Example: '000000.pcd' -> '000000.bin'
    """
    return pcd_filename.replace('.pcd', '.bin')

def get_label_filename_from_pcd(pcd_filename):
    """
    Convert PCD filename to corresponding LABEL filename  
    Example: '000000.pcd' -> '000000.label'
    """
    return pcd_filename.replace('.pcd', '.label')

def process_single_frame(json_data, pcd_filename, bin_folder, output_folder):
    """
    Process a single frame (pcd file) from the JSON data
    
    Args:
        json_data: Complete JSON data loaded from file
        pcd_filename: Name of the PCD file (e.g., '000000.pcd')
        bin_folder: Path to folder containing .bin files
        output_folder: Path to output folder for .label files
    
    Returns:
        bool: Success status
    """
    # Generate corresponding filenames
    bin_filename = get_bin_filename_from_pcd(pcd_filename)
    label_filename = get_label_filename_from_pcd(pcd_filename)
    
    bin_file = Path(bin_folder) / bin_filename
    output_label_file = Path(output_folder) / label_filename
    
    # Check if corresponding bin file exists
    if not bin_file.exists():
        print(f"Warning: No corresponding .bin file found for {pcd_filename}")
        return False
    
    # Load point cloud
    points = load_point_cloud(bin_file)
    if points is None:
        return False
    
    num_points = points.shape[0]
    
    # Initialize all points as static (label = 251)
    # SemanticKITTI-MOS format: 251 = static, 9 = moving
    labels = np.full(num_points, 251, dtype=np.uint32)
    
    # Process annotations for this specific frame
    if 'labels' in json_data and pcd_filename in json_data['labels']:
        annotations = json_data['labels'][pcd_filename]
        
        moving_points_total = 0
        
        for annotation in annotations:
            # Extract bounding box information
            box_info = annotation.get('box', {})
            cx = box_info.get('cx', 0)
            cy = box_info.get('cy', 0) 
            cz = box_info.get('cz', 0)
            l = box_info.get('l', 0)  # length
            w = box_info.get('w', 0)  # width  
            h = box_info.get('h', 0)  # height
            rot_z = box_info.get('rot_z', 0)
            
            # Extract motion-related information
            action_labels = annotation.get('action_label', {})
            social_activity = annotation.get('social_activity', {})
            
            # Determine if object is moving
            is_moving = determine_motion_state(action_labels, social_activity)
            
            # Find points inside the bounding box
            box_center = [cx, cy, cz]
            box_dimensions = [l, w, h]
            
            inside_box = point_in_3d_box(points, box_center, box_dimensions, rot_z)
            points_in_box = np.sum(inside_box)
            
            # Assign motion labels to points inside the box
            if is_moving:
                labels[inside_box] = 9  # Moving object
                moving_points_total += points_in_box
        
        print(f"  {pcd_filename}: {num_points} points total, {moving_points_total} moving, {num_points - moving_points_total} static")
    else:
        print(f"  {pcd_filename}: No annotations found, all points marked as static")
    
    # Save labels in binary format (uint32, little endian)
    try:
        with open(output_label_file, 'wb') as f:
            for label in labels:
                f.write(struct.pack('<I', label))  # '<I' = little-endian unsigned int (32-bit)
        
        return True
        
    except Exception as e:
        print(f"Error saving label file {output_label_file}: {e}")
        return False

def convert_jrdb_sequence_json_to_mos(json_file, bin_folder, output_folder):
    """
    Convert single JRDB JSON file (containing sequence data) to multiple KITTI MOS label files
    
    Args:
        json_file: Path to JRDB JSON annotation file (single file for entire sequence)
        bin_folder: Path to folder containing .bin point cloud files  
        output_folder: Path to output folder for .label files
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Load JSON data
    try:
        with open(json_file, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file}: {e}")
        return
    
    # Extract all PCD filenames from JSON labels
    if 'labels' not in json_data:
        print(f"Error: No 'labels' key found in JSON file {json_file}")
        return
    
    pcd_filenames = list(json_data['labels'].keys())
    pcd_filenames.sort()  # Sort to ensure consistent processing order
    
    print(f"Found {len(pcd_filenames)} frames in JSON file")
    
    # Get list of available bin files for verification
    bin_files = list(Path(bin_folder).glob('*.bin'))
    bin_names = {bf.name for bf in bin_files}
    
    print(f"Found {len(bin_files)} .bin files in {bin_folder}")
    
    successful_conversions = 0
    missing_bin_files = []
    
    # Process each frame
    for pcd_filename in tqdm(pcd_filenames, desc="Converting frames to MOS labels"):
        # Check if corresponding bin file exists
        expected_bin_name = get_bin_filename_from_pcd(pcd_filename)
        
        if expected_bin_name not in bin_names:
            missing_bin_files.append(expected_bin_name)
            continue
        
        # Process this frame
        if process_single_frame(json_data, pcd_filename, bin_folder, output_folder):
            successful_conversions += 1
    
    print(f"\nConversion completed:")
    print(f"  Successfully converted: {successful_conversions}/{len(pcd_filenames)} frames")
    
    if missing_bin_files:
        print(f"  Missing .bin files: {len(missing_bin_files)}")
        if len(missing_bin_files) <= 10:  # Show first 10 missing files
            print(f"  Missing files: {missing_bin_files}")
        else:
            print(f"  First 10 missing files: {missing_bin_files[:10]}")

def verify_label_file(label_file, bin_file):
    """
    Verify that the label file has correct format and matches point cloud
    """
    try:
        # Load point cloud to get number of points
        points = load_point_cloud(bin_file)
        if points is None:
            return False
        
        num_points = points.shape[0]
        
        # Load labels
        with open(label_file, 'rb') as f:
            label_data = f.read()
        
        # Check if label file size matches expected size (4 bytes per point)
        expected_size = num_points * 4
        actual_size = len(label_data)
        
        if actual_size != expected_size:
            print(f"Error: Label file size mismatch. Expected: {expected_size}, Got: {actual_size}")
            return False
        
        # Parse labels
        labels = np.frombuffer(label_data, dtype=np.uint32)
        
        # Check label values
        unique_labels = np.unique(labels)
        valid_labels = {9, 251}  # MOS labels: 9=moving, 251=static
        
        invalid_labels = set(unique_labels) - valid_labels
        if invalid_labels:
            print(f"Warning: Found invalid labels: {invalid_labels}")
        
        # Statistics
        moving_count = np.sum(labels == 9)
        static_count = np.sum(labels == 251)
        
        print(f"Label file verification successful:")
        print(f"  Points: {num_points}")
        print(f"  Moving: {moving_count} ({moving_count/num_points*100:.1f}%)")
        print(f"  Static: {static_count} ({static_count/num_points*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"Error verifying label file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert single JRDB sequence JSON to multiple KITTI MOS label files")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to single JRDB JSON annotation file (contains entire sequence)")
    parser.add_argument("--bin_folder", type=str, required=True,
                        help="Path to folder containing .bin point cloud files (velodyne folder)")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to output folder for .label files")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the first converted label file")
    
    args = parser.parse_args()
    
    # Check input files/directories
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file {args.json_file} does not exist")
        return
    
    if not os.path.exists(args.bin_folder):
        print(f"Error: Bin folder {args.bin_folder} does not exist")
        return
    
    print(f"Processing JSON file: {args.json_file}")
    print(f"Bin folder: {args.bin_folder}")
    print(f"Output folder: {args.output_folder}")
    
    # Convert files
    convert_jrdb_sequence_json_to_mos(args.json_file, args.bin_folder, args.output_folder)
    
    # Verify first file if requested
    if args.verify:
        label_files = list(Path(args.output_folder).glob('*.label'))
        if label_files:
            first_label = sorted(label_files)[0]
            bin_name = first_label.stem + '.bin'
            corresponding_bin = Path(args.bin_folder) / bin_name
            
            if corresponding_bin.exists():
                print(f"\nVerifying {first_label.name}:")
                verify_label_file(first_label, corresponding_bin)
            else:
                print(f"Cannot verify: corresponding bin file {bin_name} not found")

if __name__ == "__main__":
    main()
