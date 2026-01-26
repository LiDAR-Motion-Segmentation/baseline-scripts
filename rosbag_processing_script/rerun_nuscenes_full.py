"""
viz_nuscenes_semseg.py

Features:
- Ego-Vehicle Filtering (Clean Center)
- Fixed 3D Box Rotation
- Toggles for Cameras, 2D Labels, and Semantic Segmentation

Usage:
    # 1. Clean Geometry (Green Points)
    python viz_nuscenes_semseg.py --root /data/nuscenes --scene 0 --show-images

    # 2. Semantic Segmentation (Colored Points)
    python viz_nuscenes_semseg.py --root /data/nuscenes --scene 0 --show-images --show-semseg
"""

import os
import argparse
import numpy as np
import cv2
import rerun as rr
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

# --- CONFIGURATION ---
CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# --- HELPER FUNCTIONS ---


def filter_ego_points(points, labels=None):
    """
    Removes points (and optional labels) that hit the ego vehicle.
    Crop Box: X[-1.5, 2.5], Y[-1.0, 1.0]
    """
    mask = ~(
        (points[:, 0] > -1.5)
        & (points[:, 0] < 2.5)
        & (points[:, 1] > -1.0)
        & (points[:, 1] < 1.0)
    )

    filtered_points = points[mask]
    filtered_labels = labels[mask] if labels is not None else None

    return filtered_points, filtered_labels


def load_lidarseg(nusc, sample_token):
    """Load per-point segmentation labels (LidarSeg)."""
    # try:
    lidarseg_token = nusc.get("sample", sample_token)["data"]["LIDAR_TOP"]
    if lidarseg_token not in nusc.lidarseg:
        return None

    lidarseg_rec = nusc.get("lidarseg", lidarseg_token)
    lidarseg_file = os.path.join(nusc.dataroot, lidarseg_rec["filename"])
    return np.fromfile(lidarseg_file, dtype=np.uint8)
    # except:
    #     return None


def get_boxes_3d(nusc, sample_data_token):
    """Retrieve 3D bounding boxes with corrected orientation."""
    _, boxes, _ = nusc.get_sample_data(
        sample_data_token, use_flat_vehicle_coordinates=False
    )

    centers, half_sizes, rotations, labels, class_ids = [], [], [], [], []

    for box in boxes:
        centers.append(box.center)
        # FIX: Swap Width (0) and Length (1)
        w, l, h = box.wlh
        half_sizes.append([l / 2.0, w / 2.0, h / 2.0])

        # FIX: Quaternion [w, x, y, z] -> [x, y, z, w]
        q = box.orientation
        rotations.append([q[1], q[2], q[3], q[0]])

        labels.append(box.name)
        class_ids.append(abs(hash(box.name)) % 255)

    return centers, half_sizes, rotations, class_ids, labels


def get_boxes_2d(nusc, cam_token):
    """Project 3D boxes onto 2D camera images."""
    _, boxes, cam_intrinsic = nusc.get_sample_data(
        cam_token, use_flat_vehicle_coordinates=False
    )
    mins, sizes, labels, class_ids = [], [], [], []

    for box in boxes:
        if box.center[2] < 0:
            continue

        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]

        min_x, max_x = np.min(corners_2d[0]), np.max(corners_2d[0])
        min_y, max_y = np.min(corners_2d[1]), np.max(corners_2d[1])

        if max_x < 0 or min_x > 1600 or max_y < 0 or min_y > 900:
            continue

        mins.append([min_x, min_y])
        sizes.append([max_x - min_x, max_y - min_y])
        labels.append(box.name)
        class_ids.append(abs(hash(box.name)) % 255)

    return mins, sizes, labels, class_ids


def log_cameras(nusc, sample, show_2d_labels=False):
    """Log camera images and optional 2D boxes."""
    for cam_name in CAMERAS:
        if cam_name not in sample["data"]:
            continue
        cam_token = sample["data"][cam_name]
        cam_data = nusc.get("sample_data", cam_token)

        img_path = os.path.join(nusc.dataroot, cam_data["filename"])
        if os.path.exists(img_path):
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            rr.log(f"world/cameras/{cam_name}", rr.Image(img))

            if show_2d_labels:
                mins, sizes, lbls, ids = get_boxes_2d(nusc, cam_token)
                if mins:
                    rr.log(
                        f"world/cameras/{cam_name}/boxes",
                        rr.Boxes2D(mins=mins, sizes=sizes, labels=lbls, class_ids=ids),
                    )


# --- MAIN LOOP ---


def run_visualization(args):
    if not os.path.exists(args.root):
        print(f"[Error] Path not found: {args.root}")
        return

    print(f"[Info] Loading NuScenes from {args.root}...")
    version = "v1.0-mini" if "mini" in args.root else "v1.0-trainval"
    try:
        nusc = NuScenes(version=version, dataroot=args.root, verbose=False)
    except Exception as e:
        print(f"[Error] Failed to load NuScenes: {e}")
        return

    rr.init("NuScenes SemSeg Viz", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    # --- SETUP COLORS ---
    # We map LidarSeg class IDs to Rerun colors
    class_desc = []
    if hasattr(nusc, "lidarseg_idx2name_mapping"):
        for i, name in nusc.lidarseg_idx2name_mapping.items():
            class_desc.append((i, name, nusc.colormap[name]))
    if class_desc:
        rr.log("world", rr.AnnotationContext(class_desc), static=True)

    # --- STREAM ---
    if args.scene >= len(nusc.scene):
        print(f"[Error] Scene {args.scene} out of bounds.")
        return

    scene = nusc.scene[args.scene]
    token = scene["first_sample_token"]
    frame_idx = 0

    print(f"[Info] Streaming Scene {args.scene}: {scene['name']}")
    print(f"       Mode: SemSeg={'ON' if args.show_semseg else 'OFF'} (Green Mode)")

    while token != "":
        rr.set_time_sequence("frame_idx", frame_idx)

        sample = nusc.get("sample", token)
        lidar_token = sample["data"]["LIDAR_TOP"]

        # 1. Load Point Cloud
        lidar_data = nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_data["filename"])
        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points.T[:, :3]

        # 2. Load Segmentation (Optional)
        seg_labels = None
        if args.show_semseg:
            seg_labels = load_lidarseg(nusc, token)

        # 3. Filter Ego Vehicle (Applied to BOTH points and labels)
        points, seg_labels = filter_ego_points(points, seg_labels)

        # 4. Log LiDAR
        if args.show_semseg and seg_labels is not None:
            # Semantic Mode
            rr.log("world/lidar", rr.Points3D(points, class_ids=seg_labels))
        else:
            # Clean Geometry Mode (Green)
            rr.log("world/lidar", rr.Points3D(points, colors=[0, 255, 0], radii=0.08))

        # 5. Log 3D Boxes
        c, h, r, ids, lbls = get_boxes_3d(nusc, lidar_token)
        if c:
            rr.log(
                "world/boxes",
                rr.Boxes3D(
                    centers=c, half_sizes=h, quaternions=r, class_ids=ids, labels=lbls
                ),
            )

        # 6. Log Cameras
        if args.show_images:
            log_cameras(nusc, sample, show_2d_labels=args.show_2d)

        token = sample["next"]
        frame_idx += 1

    print("[Success] Visualization Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="NuScenes root path")
    parser.add_argument("--scene", type=int, default=0)

    # Toggles
    parser.add_argument("--show-images", action="store_true", help="Show Camera Feeds")
    parser.add_argument(
        "--show-2d", action="store_true", help="Show 2D Boxes on Images"
    )
    parser.add_argument(
        "--show-semseg", action="store_true", help="Enable Semantic Segmentation Colors"
    )

    args = parser.parse_args()
    if args.show_2d and not args.show_images:
        args.show_images = True

    run_visualization(args)
