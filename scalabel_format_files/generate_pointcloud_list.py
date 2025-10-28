#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path


def generate_pointcloud_list(
    input_folder,
    output_file="point_cloud_list.yml",
    base_url="http://localhost:8686/img",
):
    """
    Generate YAML file listing all .ply files in the folder

    Args:
        input_folder: Path to folder containing .ply files
        output_file: Output YAML file path
        base_url: Base URL for the point cloud files
    """
    ply_pattern = os.path.join(input_folder, "*.ply")
    ply_files = glob.glob(ply_pattern)

    if not ply_files:
        print(f"No .ply files found in {input_folder}")
        return

    def extract_frame_id(filepath):
        filename = os.path.basename(filepath)
        stem = os.path.splitext(filename)[0]
        try:
            return int(stem)
        except ValueError:
            return None

    ply_files.sort(key=extract_frame_id)
    print(f"Found {len(ply_files)} .ply files")

    yaml_lines = []
    for ply_file in ply_files:
        filename = os.path.basename(ply_file)
        yaml_entry = f'- {{\n    url: "{base_url}/{filename}",\n  }}'
        yaml_lines.append(yaml_entry)

    output_path = Path(output_file)

    with open(output_path, "w") as f:
        f.write("\n".join(yaml_lines))
        f.write("\n")  # Add final newline

    print(f"Generated {output_file} with {len(ply_files)} entries")
    print(f"Output saved to: {output_path.absolute()}")

    print("\nPreview (first 3 entries):")
    for i, line in enumerate(yaml_lines[:3]):
        print(line)
        if i < 2:  # Add blank line between entries except the last one
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML file listing all.ply files in a folder"
    )
    parser.add_argument(
        "input_folder", type=str, help="Path to folder containing.ply files"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="point_cloud_list.yml",
        help="Output YAML file path",
    )
    parser.add_argument(
        "-b",
        "--base_url",
        type=str,
        default="http://localhost:8686/img",
        help="Base URL for the point cloud files",
    )

    args = parser.parse_args()

    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        return 1

    args = parser.parse_args()
    generate_pointcloud_list(args.input_folder, args.output, args.base_url)
    return 0


if __name__ == "__main__":
    exit(main())
