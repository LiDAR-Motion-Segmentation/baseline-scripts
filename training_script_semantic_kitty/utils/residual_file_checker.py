#!/usr/bin/env python3
"""
Debug script to check residual file counts across sequences
"""
import os
import glob

def check_residual_files(data_root):
    """Check consistency of residual files across sequences"""
    sequences = sorted(glob.glob(os.path.join(data_root, "sequences", "*")))
    
    for seq_path in sequences:
        seq_name = os.path.basename(seq_path)
        print(f"\nSequence {seq_name}:")
        
        # Check each residual folder
        residual_counts = {}
        for i in range(1, 9):
            residual_folder = os.path.join(seq_path, f"residual_images_{i}")
            if os.path.exists(residual_folder):
                files = sorted(glob.glob(os.path.join(residual_folder, "*.npy")))
                residual_counts[i] = len(files)
                print(f"  residual_images_{i}: {len(files)} files")
            else:
                print(f"  residual_images_{i}: MISSING")
                residual_counts[i] = 0
        
        # Check for consistency
        counts = list(residual_counts.values())
        if len(set(counts)) > 1:
            print(f"  WARNING: Inconsistent file counts: {residual_counts}")
        
        # Check against velodyne folder
        velodyne_folder = os.path.join(seq_path, "velodyne")
        if os.path.exists(velodyne_folder):
            velodyne_files = sorted(glob.glob(os.path.join(velodyne_folder, "*.bin")))
            print(f"  velodyne: {len(velodyne_files)} files")
            
            # Compare with residual files
            if residual_counts[1] != len(velodyne_files):
                print(f"  ERROR: Velodyne files ({len(velodyne_files)}) != residual files ({residual_counts[1]})")

if __name__ == "__main__":
    # Replace with your actual data root
    data_root = "/scratch/soumo_roy/semantic-kitty-dataset/dataset"
    check_residual_files(data_root)
