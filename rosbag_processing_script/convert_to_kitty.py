#!/usr/bin/env python3
import numpy as np
import shutil
from pathlib import Path
import json
import argparse

def convert_to_kitti_format(annotations_dir, output_dir):
    output_dir = Path(output_dir)
    
    # creating kitty sequence structure
    # remember to change each sequence number for each bag
    sequences_dir = output_dir / "sequences" / "00"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    
    (sequences_dir / "velodyne").mkdir(exist_ok=True)
    (sequences_dir / "labels").mkdir(exist_ok=True)
    (sequences_dir / "image_2").mkdir(exist_ok=True)
    (sequences_dir / "image_3").mkdir(exist_ok=True)
    
    annotations_dir = Path(annotations_dir)
    
    # copy and renaming files
    label_files = sorted(list((annotations_dir / "labels").glob(".label")))
    
    for i, label_files in enumerate(label_files):
        timstamp = label_files.stem
        
        # Copy label file with sequential naming
        target_label = sequences_dir / "labels" / f"{i:06d}.label"
        shutil.copy2(label_files, target_label)
        print(f"Converted {i+1}/{len(label_files)} files", end='\r')
        
        print(f"\n Converted {len(label_files)} files to KITTI format")
        print(f" KITTI dataset: {output_dir}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_dir")
    parser.add_argument("--output", "-o", default="kitty_dataset")
    args = parser.parse_args()
    convert_to_kitti_format(args.annotation_dir, args.output)