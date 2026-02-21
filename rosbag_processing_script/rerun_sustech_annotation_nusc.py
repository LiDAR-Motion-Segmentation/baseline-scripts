"""
Visualizes .pcd point clouds, .json 3D bounding boxes, and 6 camera views in Rerun.

Usage:
    python rerun_sustech_annotation_nusc.py \
        --pcd_dir ./data/pcds \
        --json_dir ./data/labels \
        --img_dir ./data/images
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import open3d as o3d
import rerun as rr
from scipy.spatial.transform import Rotation as R

# 1. Define classes and colors based on user specification
CLASS_INFO = [
    (0, "unlabeled", [128, 128, 128]),  # Fallback
    (1, "moving_people", [255, 0, 0]),  # Bright Red
    (2, "static_people", [0, 255, 0]),  # Green
    (3, "unknown", [0, 255, 0]),  # Green
    (4, "static_car", [255, 255, 255]),  # White
    (5, "moving_car", [255, 0, 255]),  # Magenta
    (6, "moving_truck", [255, 140, 0]),  # Dark Orange
    (7, "static_truck", [255, 20, 147]),  # Deep Pink
    (8, "moving_bus", [255, 215, 0]),  # Gold
    (9, "static_bus", [128, 128, 0]),  # Olive
    (10, "moving_cyclist", [0, 255, 255]),  # Cyan
    (11, "static_cyclist", [102, 0, 204]),  # Purple
    (12, "moving_construction_vehicle", [255, 20, 147]),  # Deep Pink
    (13, "static_construction_vehicle", [199, 21, 133]),  # Medium Violet Red
    (14, "moving_other_vehicle", [138, 43, 226]),  # Blue-Violet
    (15, "static_other_vehicle", [75, 0, 130]),  # Indigo
]

# Create lookup dictionary for parsing the JSON
CLASS_MAP = {name: cls_id for cls_id, name, color in CLASS_INFO}

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def load_bounding_boxes(json_path):
    if not os.path.exists(json_path):
        return None, None, None, None, None

    with open(json_path, "r") as f:
        data = json.load(f)

    centers, half_sizes, quaternions, class_ids, labels = [], [], [], [], []

    for obj in data:
        # Position
        pos = obj["psr"]["position"]
        centers.append([pos["x"], pos["y"], pos["z"]])

        # Scale -> Half Sizes
        scale = obj["psr"]["scale"]
        half_sizes.append([scale["x"] / 2.0, scale["y"] / 2.0, scale["z"] / 2.0])

        # Rotation -> Quaternion [x, y, z, w]
        rot = obj["psr"]["rotation"]
        r = R.from_euler("xyz", [rot["x"], rot["y"], rot["z"]])
        quaternions.append(r.as_quat())

        # Classification
        obj_type = obj["obj_type"]
        obj_id = obj["obj_id"]
        labels.append(f"{obj_type} (ID: {obj_id})")

        # Default to 0 (unlabeled) if the class name isn't in our list
        class_ids.append(CLASS_MAP.get(obj_type, 0))

    return centers, half_sizes, quaternions, class_ids, labels


def log_cameras(img_dir, base_name):
    """
    Looks for images across the 6 camera folders and logs them to Rerun.
    """
    if not img_dir or not os.path.exists(img_dir):
        return

    for cam in CAMERAS:
        cam_dir = os.path.join(img_dir, cam)

        # Check for common image extensions
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            test_path = os.path.join(cam_dir, f"{base_name}{ext}")
            if os.path.exists(test_path):
                img_path = test_path
                break

        if img_path:
            img_bgr = cv2.imread(img_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                rr.log(f"world/cameras/{cam}", rr.Image(img_rgb))


def main(args):
    # Initialize Rerun
    rr.init("PCD + Cameras Visualizer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world", rr.AnnotationContext(CLASS_INFO), static=True)

    # Get files
    pcd_files = glob.glob(os.path.join(args.pcd_dir, "*.pcd"))
    pcd_files.sort()

    if not pcd_files:
        print(f"[Error] No .pcd files found in {args.pcd_dir}")
        return

    print(f"[Info] Found {len(pcd_files)} frames. Streaming to Rerun...")

    # Stream
    for pcd_path in pcd_files:
        base_name = os.path.splitext(os.path.basename(pcd_path))[0]

        try:
            frame_idx = int(base_name)
        except ValueError:
            frame_idx += 1

        rr.set_time_sequence("frame_idx", frame_idx)

        # A. Log Point Cloud
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)

        if len(points) > 0:
            # Render points dimly so bright bounding boxes stand out
            rr.log(
                "world/lidar", rr.Points3D(points, colors=[150, 150, 150], radii=0.05)
            )

        # B. Log Bounding Boxes
        json_path = os.path.join(args.json_dir, f"{base_name}.json")
        centers, half_sizes, quaternions, class_ids, labels = load_bounding_boxes(
            json_path
        )

        if centers:
            rr.log(
                "world/boxes",
                rr.Boxes3D(
                    centers=centers,
                    half_sizes=half_sizes,
                    quaternions=quaternions,
                    class_ids=class_ids,
                    labels=labels,
                ),
            )

        # C. Log Cameras
        if args.img_dir:
            log_cameras(args.img_dir, base_name)

    print("[Success] Visualization stream complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize .pcd, .json boxes, and images in Rerun"
    )
    parser.add_argument(
        "--pcd_dir", type=str, required=True, help="Directory with .pcd files"
    )
    parser.add_argument(
        "--json_dir", type=str, required=True, help="Directory with .json boxes"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Root directory containing CAM_FRONT, CAM_BACK folders etc.",
    )

    args = parser.parse_args()
    main(args)
