#!/usr/bin/env python3
"""
Generate KITTI format times.txt file from JRDB dataset
Converts JRDB point cloud sequence to KITTI-compatible timestamp format
python3 times_txt_file_generator.py /scratch/soumo_roy/JRDB/pointclouds/upper_velodyne/bytes-cafe-2019-02-07_0/
 --output /scratch/soumo_roy/JRDB/kitty-format/sequences/00/times.txt --verify
"""

import os
import numpy as np
import argparse
from pathlib import Path
import re
from typing import List, Tuple, Optional

def extract_frame_numbers_from_filenames(file_list: List[str]) -> List[int]:
    """
    Extract frame numbers from JRDB filename format
    Expected formats: 000000.pcd, 000000.bin, etc.
    
    Args:
        file_list: List of filenames
    
    Returns:
        List of frame numbers (integers)
    """
    frame_numbers = []
    
    for filename in file_list:
        # Remove extension and extract numeric part
        base_name = os.path.splitext(filename)[0]
        
        # Extract digits from filename (handle various naming conventions)
        match = re.search(r'(\d+)', base_name)
        if match:
            frame_numbers.append(int(match.group(1)))
        else:
            print(f"Warning: Could not extract frame number from {filename}")
    
    return frame_numbers

def calculate_timestamps_from_fps(num_frames: int, fps: float = 15.0) -> np.ndarray:
    """
    Calculate timestamps based on JRDB's frame rate
    
    Args:
        num_frames: Total number of frames
        fps: Frames per second (default: 15.0 for JRDB)
    
    Returns:
        Array of timestamps in seconds
    """
    # Calculate time interval between frames
    dt = 1.0 / fps
    
    # Generate timestamps starting from 0
    timestamps = np.arange(num_frames, dtype=np.float64) * dt
    
    return timestamps

def estimate_fps_from_sequence_duration(num_frames: int, estimated_duration: Optional[float] = None) -> float:
    """
    Estimate FPS if sequence duration is known
    
    Args:
        num_frames: Total number of frames
        estimated_duration: Estimated duration in seconds (optional)
    
    Returns:
        Estimated FPS
    """
    if estimated_duration is None:
        # Default to JRDB's documented 15 fps
        return 15.0
    
    return num_frames / estimated_duration

def generate_timestamps_from_frame_intervals(frame_numbers: List[int], fps: float = 15.0) -> np.ndarray:
    """
    Generate timestamps considering potential frame gaps in sequence
    
    Args:
        frame_numbers: List of frame numbers (may have gaps)
        fps: Frames per second
    
    Returns:
        Array of timestamps corresponding to frame numbers
    """
    # Sort frame numbers to ensure correct order
    sorted_frames = sorted(frame_numbers)
    
    # Calculate time interval
    dt = 1.0 / fps
    
    # Generate timestamps based on frame numbers
    # Assumes frame 0 corresponds to time 0
    min_frame = min(sorted_frames)
    timestamps = np.array([(frame - min_frame) * dt for frame in sorted_frames], dtype=np.float64)
    
    return timestamps

def detect_jrdb_file_structure(input_directory: str) -> Tuple[List[str], str]:
    """
    Detect JRDB file structure and return list of point cloud files
    
    Args:
        input_directory: Path to JRDB directory
    
    Returns:
        Tuple of (file_list, file_extension)
    """
    input_path = Path(input_directory)
    
    # Check for common JRDB point cloud file extensions
    extensions_to_check = ['.pcd', '.bin', '.ply']
    
    for ext in extensions_to_check:
        files = list(input_path.glob(f'*{ext}'))
        if files:
            file_names = [f.name for f in sorted(files)]
            print(f"Found {len(files)} {ext} files")
            return file_names, ext
    
    # If no point cloud files found, check subdirectories
    for subdir in ['pointclouds', 'velodyne', 'upper_velodyne', 'lower_velodyne']:
        subdir_path = input_path / subdir
        if subdir_path.exists():
            for ext in extensions_to_check:
                files = list(subdir_path.glob(f'*{ext}'))
                if files:
                    file_names = [f.name for f in sorted(files)]
                    print(f"Found {len(files)} {ext} files in {subdir}/")
                    return file_names, ext
    
    raise FileNotFoundError(f"No point cloud files found in {input_directory}")

def save_kitti_times_file(timestamps: np.ndarray, output_file: str) -> None:
    """
    Save timestamps in KITTI times.txt format
    
    Args:
        timestamps: Array of timestamps in seconds
        output_file: Path to output times.txt file
    """
    with open(output_file, 'w') as f:
        for timestamp in timestamps:
            # KITTI format: scientific notation with 6 decimal places
            f.write(f"{timestamp:.6e}\n")

def verify_times_file(times_file: str, expected_count: int) -> bool:
    """
    Verify the generated times.txt file
    
    Args:
        times_file: Path to times.txt file
        expected_count: Expected number of timestamps
    
    Returns:
        True if verification passes
    """
    try:
        with open(times_file, 'r') as f:
            lines = f.readlines()
        
        # Check number of lines
        if len(lines) != expected_count:
            print(f"Error: Expected {expected_count} timestamps, found {len(lines)}")
            return False
        
        # Verify format and monotonicity
        timestamps = []
        for i, line in enumerate(lines):
            line = line.strip()
            try:
                timestamp = float(line)
                timestamps.append(timestamp)
                
                # Verify scientific notation format
                if 'e' not in line.lower():
                    print(f"Warning: Line {i+1} not in scientific notation: {line}")
                
            except ValueError:
                print(f"Error: Invalid timestamp format at line {i+1}: {line}")
                return False
        
        # Check if timestamps are monotonically increasing
        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            print("Error: Timestamps are not monotonically increasing")
            return False
        
        # Display statistics
        print(f"Verification successful:")
        print(f"  Total timestamps: {len(timestamps)}")
        print(f"  Time range: {timestamps[0]:.6f}s to {timestamps[-1]:.6f}s")
        print(f"  Duration: {timestamps[-1] - timestamps[0]:.6f}s")
        
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            estimated_fps = 1.0 / avg_interval if avg_interval > 0 else 0
            print(f"  Average interval: {avg_interval:.6f}s")
            print(f"  Estimated FPS: {estimated_fps:.2f}")
        
        return True
        
    except Exception as e:
        print(f"Error verifying times file: {e}")
        return False

def preview_times_file(times_file: str, num_lines: int = 10) -> None:
    """
    Preview the first few lines of the generated times.txt file
    
    Args:
        times_file: Path to times.txt file
        num_lines: Number of lines to preview
    """
    try:
        with open(times_file, 'r') as f:
            lines = f.readlines()
        
        print(f"\nPreview of {times_file} (first {min(num_lines, len(lines))} lines):")
        print("-" * 50)
        for i, line in enumerate(lines[:num_lines]):
            print(f"{i:06d}: {line.strip()}")
        
        if len(lines) > num_lines:
            print(f"... ({len(lines) - num_lines} more lines)")
        
    except Exception as e:
        print(f"Error previewing times file: {e}")

def generate_jrdb_to_kitti_times(input_directory: str, output_file: str, 
                                fps: Optional[float] = None, 
                                sequence_duration: Optional[float] = None,
                                force_fps: bool = False) -> None:
    """
    Main function to generate KITTI times.txt from JRDB dataset
    
    Args:
        input_directory: Path to JRDB directory containing point cloud files
        output_file: Path to output times.txt file
        fps: Custom FPS value (default: auto-detect or use 15.0)
        sequence_duration: Known sequence duration in seconds (for FPS estimation)
        force_fps: Whether to force the provided FPS value
    """
    
    print(f"Generating KITTI times.txt from JRDB dataset...")
    print(f"Input directory: {input_directory}")
    print(f"Output file: {output_file}")
    
    # Detect JRDB file structure
    try:
        file_list, file_extension = detect_jrdb_file_structure(input_directory)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Extract frame numbers
    frame_numbers = extract_frame_numbers_from_filenames(file_list)
    
    if not frame_numbers:
        print("Error: No valid frame numbers found in filenames")
        return
    
    num_frames = len(frame_numbers)
    print(f"Detected {num_frames} frames with {file_extension} extension")
    
    # Determine FPS
    if fps is None:
        if sequence_duration is not None:
            estimated_fps = estimate_fps_from_sequence_duration(num_frames, sequence_duration)
            print(f"Estimated FPS from duration: {estimated_fps:.2f}")
            fps = estimated_fps
        else:
            fps = 15.0  # JRDB default
            print(f"Using default JRDB FPS: {fps}")
    else:
        print(f"Using provided FPS: {fps}")
    
    # Generate timestamps
    if len(set(frame_numbers)) == len(frame_numbers) and min(frame_numbers) == 0:
        # Consecutive frames starting from 0
        timestamps = calculate_timestamps_from_fps(num_frames, fps)
        print("Generated timestamps for consecutive frames")
    else:
        # Handle potential gaps in frame sequence
        timestamps = generate_timestamps_from_frame_intervals(frame_numbers, fps)
        print("Generated timestamps considering frame number sequence")
    
    # Save times.txt file
    save_kitti_times_file(timestamps, output_file)
    
    print(f"Successfully generated {output_file}")
    print(f"Total duration: {timestamps[-1]:.6f} seconds")
    print(f"Average FPS: {num_frames / timestamps[-1]:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Generate KITTI times.txt from JRDB dataset")
    parser.add_argument("input_directory", type=str,
                        help="Path to JRDB directory containing point cloud files")
    parser.add_argument("--output", type=str, default="times.txt",
                        help="Output times.txt file path (default: times.txt)")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate in FPS (default: auto-detect or use 15.0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Known sequence duration in seconds (for FPS estimation)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the generated times.txt file")
    parser.add_argument("--preview", type=int, default=10,
                        help="Number of lines to preview (default: 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_directory):
        print(f"Error: Input directory {args.input_directory} does not exist")
        return
    
    # Generate times.txt file
    generate_jrdb_to_kitti_times(
        input_directory=args.input_directory,
        output_file=args.output,
        fps=args.fps,
        sequence_duration=args.duration
    )
    
    # Verify the generated file
    if args.verify:
        print(f"\nVerifying generated times.txt file...")
        # Count expected frames
        try:
            file_list, _ = detect_jrdb_file_structure(args.input_directory)
            expected_count = len(file_list)
            
            if verify_times_file(args.output, expected_count):
                print("✓ Verification passed")
            else:
                print("✗ Verification failed")
                
        except Exception as e:
            print(f"Error during verification: {e}")
    
    # Preview the file
    if args.preview > 0:
        preview_times_file(args.output, args.preview)

if __name__ == "__main__":
    main()
