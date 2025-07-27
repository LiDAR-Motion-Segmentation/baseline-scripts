#!/usr/bin/env python3
"""
PCD to BIN converter for KITTI Velodyne format
Converts point cloud files from PCD format to binary format used by KITTI dataset
python3 kitty_pcd_bin_formatter.py --pcd_path /scratch/soumo_roy/JRDB/pointclouds/upper_velodyne/bytes-cafe-2019-02-07_0 
--bin_path /scratch/soumo_roy/JRDB/kitty-format/sequences/00/velodyne/ --verify
"""

import os
import numpy as np
import argparse
from tqdm import tqdm
import sys

# Try importing pypcd
try:
    from pypcd import pypcd
    USE_PYPCD = True
except ImportError:
    print("pypcd not found. Installing...")
    os.system("pip install git+https://github.com/klintan/pypcd.git")
    try:
        from pypcd import pypcd
        USE_PYPCD = True
    except ImportError:
        print("Failed to install pypcd. Falling back to open3d method.")
        USE_PYPCD = False

# Alternative: using open3d
if not USE_PYPCD:
    try:
        import open3d as o3d
        USE_OPEN3D = True
    except ImportError:
        print("open3d not found. Installing...")
        os.system("pip install open3d")
        import open3d as o3d
        USE_OPEN3D = True

def convert_pcd_to_bin_pypcd(pcd_file, bin_file):
    """
    Convert PCD file to BIN using pypcd library
    KITTI format: each point is [x, y, z, intensity] as float32
    """
    try:
        # Load PCD file
        pc = pypcd.PointCloud.from_path(pcd_file)
        
        # Get number of points
        num_points = pc.pc_data.shape[0]
        
        # Create output array with 4 channels (x, y, z, intensity)
        points = np.zeros([num_points, 4], dtype=np.float32)
        
        # Extract coordinates
        points[:, 0] = pc.pc_data['x'].astype(np.float32)
        points[:, 1] = pc.pc_data['y'].astype(np.float32) 
        points[:, 2] = pc.pc_data['z'].astype(np.float32)
        
        # Handle intensity field (may have different names)
        if 'intensity' in pc.pc_data.dtype.names:
            points[:, 3] = pc.pc_data['intensity'].astype(np.float32)
        elif 'i' in pc.pc_data.dtype.names:
            points[:, 3] = pc.pc_data['i'].astype(np.float32)
        elif 'reflectance' in pc.pc_data.dtype.names:
            points[:, 3] = pc.pc_data['reflectance'].astype(np.float32)
        else:
            # If no intensity field, set to zeros
            points[:, 3] = np.zeros(num_points, dtype=np.float32)
            print(f"Warning: No intensity field found in {pcd_file}, using zeros")
        
        # Write to binary file
        with open(bin_file, 'wb') as f:
            f.write(points.tobytes())
            
        return True
        
    except Exception as e:
        print(f"Error converting {pcd_file}: {str(e)}")
        return False

def convert_pcd_to_bin_open3d(pcd_file, bin_file):
    """
    Convert PCD file to BIN using open3d library
    Note: This method only extracts XYZ coordinates, intensity is set to 0
    """
    try:
        # Load PCD file
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        # Get points as numpy array
        points_xyz = np.asarray(pcd.points, dtype=np.float32)
        
        # Add intensity channel (zeros since open3d doesn't handle intensity well)
        intensity = np.zeros((points_xyz.shape[0], 1), dtype=np.float32)
        points = np.hstack([points_xyz, intensity])
        
        # Write to binary file
        with open(bin_file, 'wb') as f:
            f.write(points.tobytes())
            
        print(f"Warning: Intensity values set to zero for {pcd_file} (open3d limitation)")
        return True
        
    except Exception as e:
        print(f"Error converting {pcd_file}: {str(e)}")
        return False

def batch_convert_pcd_to_bin(pcd_folder, bin_folder, file_prefix=""):
    """
    Convert all PCD files in a folder to BIN format
    """
    # Create output directory if it doesn't exist
    os.makedirs(bin_folder, exist_ok=True)
    
    # Find all PCD files
    pcd_files = []
    for filename in os.listdir(pcd_folder):
        if filename.lower().endswith('.pcd'):
            pcd_files.append(os.path.join(pcd_folder, filename))
    
    # Sort files to ensure consistent ordering
    pcd_files.sort()
    
    if not pcd_files:
        print(f"No PCD files found in {pcd_folder}")
        return
    
    print(f"Found {len(pcd_files)} PCD files to convert")
    
    # Convert each file
    successful_conversions = 0
    for pcd_file in tqdm(pcd_files, desc="Converting PCD to BIN"):
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(pcd_file))[0]
        if file_prefix:
            bin_filename = f"{file_prefix}_{base_name}.bin"
        else:
            bin_filename = f"{base_name}.bin"
        
        bin_file = os.path.join(bin_folder, bin_filename)
        
        # Convert using available method
        if USE_PYPCD:
            success = convert_pcd_to_bin_pypcd(pcd_file, bin_file)
        else:
            success = convert_pcd_to_bin_open3d(pcd_file, bin_file)
        
        if success:
            successful_conversions += 1
    
    print(f"Successfully converted {successful_conversions}/{len(pcd_files)} files")

def verify_bin_file(bin_file):
    """
    Verify that the BIN file is in correct KITTI format
    """
    try:
        # Read binary file
        data = np.fromfile(bin_file, dtype=np.float32)
        
        # Check if data length is divisible by 4 (x, y, z, intensity)
        if len(data) % 4 != 0:
            print(f"Warning: {bin_file} has invalid format (length not divisible by 4)")
            return False
        
        # Reshape to N x 4
        points = data.reshape(-1, 4)
        
        print(f"{bin_file}: {points.shape[0]} points")
        print(f"X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
        print(f"Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
        print(f"Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
        print(f"Intensity range: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"Error verifying {bin_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PCD files to KITTI Velodyne BIN format")
    parser.add_argument("--pcd_path", type=str, required=True,
                        help="Path to folder containing PCD files")
    parser.add_argument("--bin_path", type=str, required=True,
                        help="Path to output folder for BIN files")
    parser.add_argument("--file_prefix", type=str, default="",
                        help="Prefix for output BIN filenames")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the first converted BIN file")
    
    args = parser.parse_args()
    
    # Check if input folder exists
    if not os.path.exists(args.pcd_path):
        print(f"Error: Input folder {args.pcd_path} does not exist")
        sys.exit(1)
    
    # Convert files
    batch_convert_pcd_to_bin(args.pcd_path, args.bin_path, args.file_prefix)
    
    # Verify first file if requested
    if args.verify:
        bin_files = [f for f in os.listdir(args.bin_path) if f.endswith('.bin')]
        if bin_files:
            first_bin = os.path.join(args.bin_path, sorted(bin_files)[0])
            print(f"\nVerifying {first_bin}:")
            verify_bin_file(first_bin)

if __name__ == "__main__":
    main()
