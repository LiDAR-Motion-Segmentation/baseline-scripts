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
    center = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
    size = np.array([scale["x"], scale["y"], scale["z"]], dtype=float)

    # label text + id might change in future
    label = f"{obj_type} (ID : {obj_id})"

    rr.log(
        f"bbox/{obj_type}/{obj_type}",
        rr.Boxes3D(centers=[center], half_sizes=size / 2.0),
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
    )

    # Draw floating text label using Points3D
    label_pos = center + np.array([0, 0, size[2] / 2])
    rr.log(
        f"labels/{obj_type}/{obj_id}",
        rr.Points3D(positions=[label_pos], labels=[label]),
    )


def show_pointcloud(pointcloud_path, frame_idx):
    pc = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(pc.points)
    if len(points) == 0:
        print(f"[WARN] Empty pointcloud at {pointcloud_path}")
        return
    colors = None
    if pc.has_colors():
        colors = (np.asarray(pc.colors) * 255).astype(np.uint8)

    rr.set_time_sequence("frame_idx", frame_idx)
    rr.log("lidar/points", rr.Points3D(positions=points, colors=colors))


def visualize_sequence(pcd_dir, json_dir, fps=2):
    pcd_dir = Path(pcd_dir)
    json_dir = Path(json_dir)
    rr.init("pcd_json_viewer", spawn=True)

    pcd_files = sorted([f for f in pcd_dir.glob("*.pcd")], key=lambda p: p.stem)
    json_files = sorted([f for f in json_dir.glob("*.json")], key=lambda p: p.stem)

    if len(pcd_files) == 0 or len(json_files) == 0:
        print(f"length of pcd files found {len(pcd_files)}")
        print(f"length of json files found {len(json_files)}")
        raise RuntimeError("No PCD or JSON files found")

    print(f"Visualizing {len(pcd_files)} frames... Press Ctrl+C to stop.")

    for frame_idx, (pcd_file, json_file) in enumerate(zip(pcd_files, json_files)):
        print(f"[Frame {frame_idx}] {pcd_file.name}")
        show_pointcloud(pcd_file, frame_idx)
        with open(json_file, "r") as f:
            objs = json.load(f)

        for obj in objs:
            psr = obj["psr"]
            draw_bounding_box(psr, obj["obj_id"], obj["obj_type"])

        rr.log("frame", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    print("Visualization complete")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D bounding boxes and pointcloud from folders."
    )
    parser.add_argument(
        "--pcd_dir", required=True, help="Path to directory of PCD files"
    )
    parser.add_argument(
        "--json_dir", required=True, help="Path to directory of bounding box JSONs"
    )
    parser.add_argument(
        "--fps", type=float, default=2.0, help="Playback FPS (default=2 FPS)"
    )
    args = parser.parse_args()

    visualize_sequence(args.pcd_dir, args.json_dir, args.fps)


if __name__ == "__main__":
    main()
