#!/usr/bin/env python3
import os
import json
import rerun as rr
import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
import cv2


def draw_bounding_box(psr, obj_id, obj_type):
    pos = psr["position"]
    rot = psr["rotation"]
    scale = psr["scale"]

    # Position
    center = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)

    # Size (Fix: Convert full scale to half-sizes)
    full_size = np.array([scale["x"], scale["y"], scale["z"]], dtype=float)
    half_size = full_size / 2.0

    # Rotation (Fix: Pass quaternion to Rerun)
    # Assuming your JSON rotation is a quaternion {x, y, z, w}
    # Rerun expects [x, y, z, w]
    quaternion = [rot["x"], rot["y"], rot["z"], rot["z"]]

    label = f"{obj_type} {obj_id}"

    # Log with Unique Path (Fix: Use obj_id in path)
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


def show_pointcloud(pointcloud_path):
    pc = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pc.points)

    if len(points) == 0:
        return

    colors = None
    if pc.has_colors():
        colors = (np.asarray(pc.colors) * 255).astype(np.uint8)

    rr.log("lidar/points", rr.Points3D(positions=points, colors=colors))


def show_image(image_path):
    if not image_path.exists():
        return

    img = cv2.imread(str(image_path))
    # Convert BGR to RGB for Rerun
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rr.log("camera/image", rr.Image(img))


def visualize_sequence(pcd_dir, json_dir, img_dir, fps=2):
    pcd_dir = Path(pcd_dir)
    json_dir = Path(json_dir)
    img_dir = Path(img_dir)

    rr.init("pcd_json_viewer", spawn=True)

    # Sort files to ensure synchronization
    pcd_files = sorted([f for f in pcd_dir.glob("*.pcd")], key=lambda p: p.stem)
    json_files = sorted([f for f in json_dir.glob("*.json")], key=lambda p: p.stem)
    img_files = sorted([f for f in img_dir.glob("*.jpg")], key=lambda p: p.stem)
    # Note: Assuming .png, change to .jpg if needed

    # Basic check to warn if counts mismatch
    if not (len(pcd_files) == len(json_files) == len(img_files)):
        print(
            f"[WARN] File counts mismatch! PCD: {len(pcd_files)}, JSON: {len(json_files)}, IMG: {len(img_files)}"
        )
        # We limit to the shortest list to avoid crashing
        min_len = min(len(pcd_files), len(json_files), len(img_files))
        pcd_files = pcd_files[:min_len]
        json_files = json_files[:min_len]
        img_files = img_files[:min_len]

    if len(pcd_files) == 0:
        raise RuntimeError("No files found.")

    print(f"Visualizing {len(pcd_files)} frames...")
    active_ids_prev_frame = set()

    for frame_idx, (pcd_file, json_file, img_file) in enumerate(
        zip(pcd_files, json_files, img_files)
    ):
        print(f"[Frame {frame_idx}] {pcd_file.name} | {img_file.name}")

        # Set Global Time
        rr.set_time_sequence("frame_idx", frame_idx)

        # 1. Log Image
        show_image(img_file)

        # 2. Log Point Cloud
        show_pointcloud(pcd_file)

        # 3. Log Bounding Boxes
        with open(json_file, "r") as f:
            objs = json.load(f)

        active_ids_current_frame = set()

        for obj in objs:
            psr = obj["psr"]
            obj_id = obj["obj_id"]
            obj_type = obj["obj_type"]
            entity_path = f"bbox/{obj_type}/{obj_id}"

            draw_bounding_box(psr, obj_id, obj_type)
            active_ids_current_frame.add(entity_path)

        # Clear disappeared objects
        disappeared_ids = active_ids_prev_frame - active_ids_current_frame
        for entity_path in disappeared_ids:
            rr.log(entity_path, rr.Clear(recursive=True))

        active_ids_prev_frame = active_ids_current_frame

    print("Visualization complete")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D bounding boxes, pointcloud, and camera images."
    )
    parser.add_argument("--pcd_dir", required=True, help="Path to PCD folder")
    parser.add_argument("--json_dir", required=True, help="Path to JSON folder")
    parser.add_argument("--img_dir", required=True, help="Path to Image folder")
    parser.add_argument("--fps", type=float, default=2.0, help="Playback FPS")
    args = parser.parse_args()

    visualize_sequence(args.pcd_dir, args.json_dir, args.img_dir, args.fps)


if __name__ == "__main__":
    main()
