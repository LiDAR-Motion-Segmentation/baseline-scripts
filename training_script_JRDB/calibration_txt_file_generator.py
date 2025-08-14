#!/usr/bin/env python3
"""
Convert JRDB multi-camera YAML calibration to KITTI calib.txt format
Handles JRDB's cylindrical camera array with stitching parameters
python3 calibration_txt_file_generator.py --camera_yaml /scratch/soumo_roy/JRDB/calibration/cameras.yaml 
--output /scratch/soumo_roy/JRDB/kitty-format/sequences/00/calib.txt
"""

import numpy as np
import yaml
import argparse
import os
from pathlib import Path

def load_yaml_file(file_path):
    """
    Load YAML calibration file with error handling
    """
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return data
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None

def parse_matrix_string(matrix_str):
    """
    Parse matrix string from YAML (handles multiline string format)
    Example: "476.71 0 350.738\n0 479.505 209.532\n0 0 1"
    """
    if isinstance(matrix_str, str):
        # Split by newlines and spaces, filter out empty strings
        values = []
        for line in matrix_str.strip().split('\n'):
            values.extend([float(x) for x in line.split() if x])
        return np.array(values)
    elif isinstance(matrix_str, list):
        # Already a list of values
        return np.array(matrix_str, dtype=float)
    else:
        raise ValueError(f"Unsupported matrix format: {type(matrix_str)}")

def parse_jrdb_cameras(camera_data):
    """
    Parse JRDB camera YAML format with sensor_0, sensor_1, etc.
    
    Returns:
        dict: Dictionary of parsed camera parameters
    """
    cameras = {}
    stitching_params = camera_data.get('stitching', {})
    camera_sensors = camera_data.get('cameras', {})
    
    print(f"Found stitching parameters: radius={stitching_params.get('radius', 'N/A')}, "
          f"scalewidth={stitching_params.get('scalewidth', 'N/A')}")
    
    for sensor_id, sensor_params in camera_sensors.items():
        if not sensor_id.startswith('sensor_'):
            continue
            
        try:
            # Parse distortion coefficients
            D_str = sensor_params.get('D', '0 0 0 0 0')
            if isinstance(D_str, str):
                D = np.array([float(x) for x in D_str.split()])
            else:
                D = np.array(D_str, dtype=float)
            
            # Parse intrinsic matrix K (3x3)
            K_str = sensor_params['K']
            K_values = parse_matrix_string(K_str)
            K = K_values.reshape(3, 3)
            
            # Parse rotation matrix R (3x3)
            R_str = sensor_params['R']
            R_values = parse_matrix_string(R_str)
            R = R_values.reshape(3, 3)
            
            # Parse translation vector T
            T_str = sensor_params['T']
            if isinstance(T_str, str):
                T = np.array([float(x) for x in T_str.split()]).reshape(3, 1)
            else:
                T = np.array(T_str, dtype=float).reshape(3, 1)
            
            # Get image dimensions
            width = sensor_params.get('width', 752)
            height = sensor_params.get('height', 480)
            
            cameras[sensor_id] = {
                'K': K,
                'D': D,
                'R': R,
                'T': T,
                'width': width,
                'height': height
            }
            
            print(f"Parsed {sensor_id}: {width}x{height}, "
                  f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
                  f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
                  
        except Exception as e:
            print(f"Error parsing {sensor_id}: {e}")
            continue
    
    return cameras, stitching_params

def select_reference_camera(cameras, preference='sensor_0'):
    """
    Select reference camera for KITTI calibration
    
    Args:
        cameras: Dictionary of parsed cameras
        preference: Preferred camera ID
    
    Returns:
        dict: Reference camera parameters
    """
    # Try preferred camera first
    if preference in cameras:
        print(f"Using {preference} as reference camera")
        return cameras[preference]
    
    # Fall back to first available camera
    if cameras:
        ref_id = list(cameras.keys())[0]
        print(f"Using {ref_id} as reference camera (fallback)")
        return cameras[ref_id]
    
    raise ValueError("No cameras available for reference")

def estimate_stereo_baseline(cameras):
    """
    Estimate stereo baseline from camera positions
    Uses translation differences between cameras
    """
    if len(cameras) < 2:
        print("Warning: Less than 2 cameras, using default baseline")
        return 0.06  # Default 6cm baseline
    
    # Calculate translation distances between adjacent cameras
    sensor_ids = sorted([k for k in cameras.keys() if k.startswith('sensor_')])
    baselines = []
    
    for i in range(len(sensor_ids) - 1):
        cam1 = cameras[sensor_ids[i]]
        cam2 = cameras[sensor_ids[i + 1]]
        
        # Calculate Euclidean distance between camera centers
        T1 = cam1['T'].flatten()
        T2 = cam2['T'].flatten()
        distance = np.linalg.norm(T2 - T1)
        baselines.append(distance)
    
    # Use median baseline
    estimated_baseline = np.median(baselines) if baselines else 0.06
    print(f"Estimated stereo baseline: {estimated_baseline:.3f}m")
    
    return estimated_baseline

def convert_to_cylindrical_projection(cameras, stitching_params):
    """
    Convert individual cameras to equivalent cylindrical projection parameters
    Based on JRDB cylindrical camera model from documentation
    """
    # Extract stitching parameters
    radius = stitching_params.get('radius', 3360000)  # in micrometers or scaled units
    scale_width = stitching_params.get('scalewidth', 1831)
    
    # Get focal lengths from all cameras
    focal_lengths_x = []
    focal_lengths_y = []
    principal_points_y = []
    
    for cam_id, cam_params in cameras.items():
        K = cam_params['K']
        focal_lengths_x.append(K[0, 0])  # fx
        focal_lengths_y.append(K[1, 1])  # fy
        principal_points_y.append(K[1, 2])  # cy
    
    # Calculate median values as per JRDB documentation
    f_median = np.median(focal_lengths_y)  # Use fy for cylindrical model
    cy_median = np.median(principal_points_y)
    
    # JRDB cylindrical model parameters
    # Cylindrical image dimensions
    W_cyl = 3760  # pixels (stitched width)
    H_cyl = 480   # pixels (stitched height)
    
    # Calculate equivalent cylindrical focal length
    # Based on JRDB equation: v = f^c * Y/Z' + y0^c
    f_cylindrical = f_median
    
    # Calculate principal point offset for cylindrical model
    # Based on JRDB equation (3): y0^c = y0_hat / h_img_hat * H_img
    h_img_median = 480  # median height of individual cameras
    y0_cylindrical = (cy_median / h_img_median) * H_cyl
    
    print(f"Cylindrical model parameters:")
    print(f"  f_cylindrical: {f_cylindrical:.1f}")
    print(f"  y0_cylindrical: {y0_cylindrical:.1f}")
    print(f"  Cylindrical image size: {W_cyl}x{H_cyl}")
    
    return f_cylindrical, y0_cylindrical, W_cyl, H_cyl

def create_projection_matrix(K, R, T, baseline_x=0.0, baseline_y=0.0):
    """
    Create KITTI-style 3x4 projection matrix
    P = K * [R|T] with baseline adjustment
    """
    # Combine rotation and translation into 3x4 matrix
    RT = np.hstack([R, T])
    
    # Apply intrinsic matrix
    P = K @ RT
    
    # Add baseline for stereo (modify the translation component)
    P[0, 3] -= K[0, 0] * baseline_x  # fx * baseline_x
    P[1, 3] -= K[1, 1] * baseline_y  # fy * baseline_y
    
    return P

def generate_kitti_calibration_from_jrdb(cameras, stitching_params, stereo_baseline=None, use_cylindrical=False):
    """
    Generate KITTI-style calibration matrices from JRDB multi-camera data
    
    Args:
        cameras: Parsed camera data
        stitching_params: Stitching parameters from YAML
        stereo_baseline: Custom stereo baseline (optional)
        use_cylindrical: Whether to use cylindrical projection model
    
    Returns:
        Dictionary with P0, P1, P2, P3, Tr matrices
    """
    
    if not cameras:
        print("Error: No camera data available")
        return None
    
    # Select reference camera
    ref_cam = select_reference_camera(cameras)
    
    # Estimate or use provided baseline
    if stereo_baseline is None:
        stereo_baseline = estimate_stereo_baseline(cameras)
    
    if use_cylindrical:
        # Use cylindrical projection model
        f_cyl, y0_cyl, W_cyl, H_cyl = convert_to_cylindrical_projection(cameras, stitching_params)
        
        # Create equivalent intrinsic matrix for cylindrical model
        K_cyl = np.array([
            [f_cyl,     0, W_cyl/2],  # Use half width as cx
            [    0, f_cyl, y0_cyl ],  # Use calculated y0
            [    0,     0,       1]
        ])
        
        # Use identity rotation and zero translation for cylindrical model
        R_ref = np.eye(3)
        T_ref = np.zeros((3, 1))
        
        K = K_cyl
        R = R_ref
        T = T_ref
        
        print("Using cylindrical projection model for KITTI conversion")
        
    else:
        # Use reference camera parameters directly
        K = ref_cam['K']
        R = ref_cam['R']
        T = ref_cam['T']
        
        print("Using reference camera model for KITTI conversion")
    
    print(f"Calibration matrices using:")
    print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"  Stereo baseline: {stereo_baseline:.3f}m")
    
    # Create projection matrices
    # P0: Left grayscale camera (reference)
    P0 = create_projection_matrix(K, R, T, 0.0, 0.0)
    
    # P1: Right grayscale camera (with baseline)
    P1 = create_projection_matrix(K, R, T, stereo_baseline, 0.0)
    
    # P2: Left color camera (same as P0 for most MOS applications)
    P2 = create_projection_matrix(K, R, T, 0.0, 0.0)
    
    # P3: Right color camera (with baseline)
    P3 = create_projection_matrix(K, R, T, stereo_baseline, 0.0)
    
    # Create LiDAR to camera transformation matrix
    # For JRDB, we use the reference camera's extrinsics
    # The Tr matrix transforms from LiDAR coordinates to camera coordinates
    Tr = np.hstack([R, T])
    
    return {
        'P0': P0,
        'P1': P1, 
        'P2': P2,
        'P3': P3,
        'Tr': Tr
    }

def save_kitti_calibration(calib_matrices, output_file):
    """
    Save calibration matrices in KITTI format
    """
    with open(output_file, 'w') as f:
        # Write projection matrices (flatten 3x4 matrix to 12 values)
        f.write(f"P0: {' '.join([f'{val:.6e}' for val in calib_matrices['P0'].flatten()])}\n")
        f.write(f"P1: {' '.join([f'{val:.6e}' for val in calib_matrices['P1'].flatten()])}\n")
        f.write(f"P2: {' '.join([f'{val:.6e}' for val in calib_matrices['P2'].flatten()])}\n")
        f.write(f"P3: {' '.join([f'{val:.6e}' for val in calib_matrices['P3'].flatten()])}\n")
        
        # Write transformation matrix (flatten 3x4 matrix to 12 values)
        f.write(f"Tr: {' '.join([f'{val:.6e}' for val in calib_matrices['Tr'].flatten()])}\n")

def analyze_camera_configuration(cameras):
    """
    Analyze the camera configuration and provide insights
    """
    print("\n=== Camera Configuration Analysis ===")
    print(f"Total cameras detected: {len(cameras)}")
    
    if not cameras:
        return
    
    # Analyze focal lengths
    focal_lengths_x = [cam['K'][0,0] for cam in cameras.values()]
    focal_lengths_y = [cam['K'][1,1] for cam in cameras.values()]
    
    print(f"Focal length range (fx): {min(focal_lengths_x):.1f} - {max(focal_lengths_x):.1f}")
    print(f"Focal length range (fy): {min(focal_lengths_y):.1f} - {max(focal_lengths_y):.1f}")
    
    # Analyze principal points
    principal_x = [cam['K'][0,2] for cam in cameras.values()]
    principal_y = [cam['K'][1,2] for cam in cameras.values()]
    
    print(f"Principal point range (cx): {min(principal_x):.1f} - {max(principal_x):.1f}")
    print(f"Principal point range (cy): {min(principal_y):.1f} - {max(principal_y):.1f}")
    
    # Analyze camera positions (translations)
    translations = [cam['T'].flatten() for cam in cameras.values()]
    if translations:
        translations = np.array(translations)
        print(f"Camera position spread:")
        print(f"  X: {translations[:,0].min():.3f} to {translations[:,0].max():.3f}")
        print(f"  Y: {translations[:,1].min():.3f} to {translations[:,1].max():.3f}")
        print(f"  Z: {translations[:,2].min():.3f} to {translations[:,2].max():.3f}")

def main():
    parser = argparse.ArgumentParser(description="Convert JRDB multi-camera YAML to KITTI format")
    parser.add_argument("--camera_yaml", type=str, required=True,
                        help="Path to JRDB camera.yaml file")
    parser.add_argument("--output", type=str, default="calib.txt",
                        help="Output KITTI calibration file path")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Custom stereo baseline in meters (auto-estimated if not provided)")
    parser.add_argument("--reference_camera", type=str, default="sensor_0",
                        help="Reference camera ID (default: sensor_0)")
    parser.add_argument("--cylindrical", action="store_true",
                        help="Use cylindrical projection model instead of reference camera")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze camera configuration and print details")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.camera_yaml):
        print(f"Error: Camera YAML file {args.camera_yaml} does not exist")
        return
    
    # Load camera calibration data
    print(f"Loading JRDB camera calibration from: {args.camera_yaml}")
    camera_raw_data = load_yaml_file(args.camera_yaml)
    if camera_raw_data is None:
        return
    
    # Parse JRDB camera format
    cameras, stitching_params = parse_jrdb_cameras(camera_raw_data)
    
    if not cameras:
        print("Error: No cameras found in YAML file")
        return
    
    # Analyze camera configuration if requested
    if args.analyze or args.verbose:
        analyze_camera_configuration(cameras)
    
    # Generate KITTI calibration matrices
    print(f"\nGenerating KITTI calibration matrices...")
    calib_matrices = generate_kitti_calibration_from_jrdb(
        cameras, 
        stitching_params, 
        stereo_baseline=args.baseline,
        use_cylindrical=args.cylindrical
    )
    
    if calib_matrices is None:
        print("Error: Failed to generate calibration matrices")
        return
    
    # Save calibration file
    save_kitti_calibration(calib_matrices, args.output)
    
    print(f"\nKITTI calibration file saved to: {args.output}")
    
    # Display matrices for verification
    if args.verbose:
        print("\n=== Generated Calibration Matrices ===")
        for key, matrix in calib_matrices.items():
            print(f"{key}:")
            print(f"{matrix}")
            print()

if __name__ == "__main__":
    main()