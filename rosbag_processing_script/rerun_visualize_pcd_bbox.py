#!/usr/bin/env python3
import os
import json
import rerun as rr
import open3d as o3d
import numpy as np
from pathlib import Path
import argparse


def draw_bounding_box(psr, obj_id, obj_type):
    pos = psr["position"]
    rot = psr["rotation"]
    scale = psr["scale"]

    # 1. Position
    center = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)

    # 2. Size (Fix: Convert full scale to half-sizes)
    full_size = np.array([scale["x"], scale["y"], scale["z"]], dtype=float)
    half_size = full_size / 2.0

    # 3. Rotation (Fix: Pass quaternion to Rerun)
    # Assuming your JSON rotation is a quaternion {x, y, z, w}
    # Rerun expects [x, y, z, w]
    quaternion = [rot["x"], rot["y"], rot["z"], rot["z"]]

    label = f"{obj_type} {obj_id}"

    # 4. Log with Unique Path (Fix: Use obj_id in path)
    # This ensures multiple objects of the same type exist simultaneously
    entity_path = f"bbox/{obj_type}/{obj_id}"

    rr.log(
        entity_path,
        rr.Boxes3D(
            centers=[center],
            half_sizes=[half_size],  # Note: Rerun expects a list of half-sizes
            quaternions=[quaternion],  # Apply the rotation
            labels=[label],  # Rerun can handle labels directly on the box
        ),
        # Optional: Keep the Z-up coordinate system
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    )


def show_pointcloud(pointcloud_path, frame_idx):
    # Set the time for this frame
    rr.set_time_sequence("frame_idx", frame_idx)

    pc = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pc.points)

    if len(points) == 0:
        return

    colors = None
    if pc.has_colors():
        colors = (np.asarray(pc.colors) * 255).astype(np.uint8)

    # Log the point cloud
    rr.log("lidar/points", rr.Points3D(positions=points, colors=colors))


def visualize_sequence(pcd_dir, json_dir, fps=2):
    pcd_dir = Path(pcd_dir)
    json_dir = Path(json_dir)

    rr.init("pcd_json_viewer", spawn=True)

    pcd_files = sorted([f for f in pcd_dir.glob("*.pcd")], key=lambda p: p.stem)
    json_files = sorted([f for f in json_dir.glob("*.json")], key=lambda p: p.stem)

    if len(pcd_files) == 0 or len(json_files) == 0:
        raise RuntimeError("No PCD or JSON files found")

    print(f"Visualizing {len(pcd_files)} frames...")

    # Keep track of what IDs were visible in the last frame
    active_ids_prev_frame = set()

    for frame_idx, (pcd_file, json_file) in enumerate(zip(pcd_files, json_files)):
        print(f"[Frame {frame_idx}] {pcd_file.name}")

        # Log Point Cloud (Time is set inside this function)
        show_pointcloud(pcd_file, frame_idx)

        # Load Current Objects
        with open(json_file, "r") as f:
            objs = json.load(f)

        # Track IDs seen in this current frame
        active_ids_current_frame = set()

        for obj in objs:
            psr = obj["psr"]
            obj_id = obj["obj_id"]
            obj_type = obj["obj_type"]

            # Construct the unique path
            entity_path = f"bbox/{obj_type}/{obj_id}"

            # Draw the box
            draw_bounding_box(psr, obj_id, obj_type)

            # Add to set
            active_ids_current_frame.add(entity_path)

        # If it was active LAST frame, but is NOT active THIS frame -> Clear it
        disappeared_ids = active_ids_prev_frame - active_ids_current_frame

        for entity_path in disappeared_ids:
            # recursive=False is usually safer if you have children (like labels attached)
            # but usually you want to wipe the whole tree for that object.
            rr.log(entity_path, rr.Clear(recursive=True))

        # Update "Previous" for the next loop
        active_ids_prev_frame = active_ids_current_frame

    print("Visualization complete")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D bounding boxes and pointcloud."
    )
    parser.add_argument("--pcd_dir", required=True, help="Path to PCD folder")
    parser.add_argument("--json_dir", required=True, help="Path to JSON folder")
    parser.add_argument("--fps", type=float, default=2.0, help="Playback FPS")
    args = parser.parse_args()

    visualize_sequence(args.pcd_dir, args.json_dir, args.fps)


if __name__ == "__main__":
    main()
