"""
viz_nuscenes_rerun.py
Visualize NuScenes LiDAR data using Rerun SDK with a timeline slider.
"""

import os
import argparse
import numpy as np
import rerun as rr
from nuscenes.nuscenes import NuScenes


def load_point_cloud(nusc, sample_token):
    """
    Helper to load binary LiDAR data for a specific sample token.
    """
    # 1. Get the data token for LIDAR_TOP
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]

    # 2. Get filepath
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_filepath = os.path.join(nusc.dataroot, lidar_data["filename"])

    # 3. Load binary (N x 5 float32: x, y, z, intensity, ring_index)
    point_cloud = np.fromfile(lidar_filepath, dtype=np.float32).reshape(-1, 5)

    # Return XYZ and Intensity
    return point_cloud[:, :3], point_cloud[:, 3]


def visualize_scene(dataroot, scene_idx=0):
    # --- 1. SETUP ---
    print("Initializing NuScenes...")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)

    # Initialize Rerun
    # spawn=True opens the viewer automatically
    rr.init("NuScenes LiDAR Viz", spawn=True)

    # --- 2. GET SCENE ---
    if scene_idx >= len(nusc.scene):
        print(f"Error: Scene index {scene_idx} out of bounds.")
        return

    my_scene = nusc.scene[scene_idx]
    first_token = my_scene["first_sample_token"]

    print(f"Streaming Scene {scene_idx}: {my_scene['name']}...")
    print("Check the Rerun window for the timeline slider.")

    # --- 3. TIMELINE LOOP ---
    current_token = first_token
    frame_idx = 0

    while current_token != "":
        # A. Load Data
        points, intensity = load_point_cloud(nusc, current_token)

        # B. Set Timeline (The "Slider")
        # This tells Rerun that everything logged below belongs to this frame index
        rr.set_time_sequence("frame_idx", frame_idx)

        # C. Colorize based on Intensity
        # Normalize intensity (0-255) to 0-1 for coloring
        normalized_intensity = np.clip(intensity / 255.0, 0, 1)
        # Create a simple color map (White = High intensity, Blue-ish = Low)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = normalized_intensity  # R
        colors[:, 1] = np.ones_like(intensity)  # G
        colors[:, 2] = normalized_intensity  # B (Tint blue)

        # D. Log to Rerun
        rr.log("world/lidar", rr.Points3D(points, colors=colors, radii=0.05))

        # E. Advance to next frame
        # NuScenes samples are a linked list
        sample = nusc.get("sample", current_token)
        current_token = sample["next"]
        frame_idx += 1

    print("Done! You can now use the slider in the Rerun viewer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize NuScenes with Rerun slider")
    parser.add_argument(
        "--root", required=True, help="Path to NuScenes root (containing v1.0-mini)"
    )
    parser.add_argument("--scene", type=int, default=0, help="Scene index to visualize")

    args = parser.parse_args()

    if os.path.exists(args.root):
        visualize_scene(args.root, args.scene)
    else:
        print(f"Path not found: {args.root}")
