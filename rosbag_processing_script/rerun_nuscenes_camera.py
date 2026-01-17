"""
viz_nuscenes_fixed_rotation.py
Visualizes NuScenes with:
1. Corrected 3D Box Orientation (Swapped W/L)
2. 2D Bounding Boxes projected onto Camera Images
3. Green Point Cloud

Usage: python viz_nuscenes_fixed_rotation.py --root /data/sets/nuscenes --scene 0
"""

import os
import argparse
import numpy as np
import cv2
import rerun as rr
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points


def get_boxes_3d(nusc, sample_data_token):
    """
    Retrieve 3D bounding boxes in Lidar frame with corrected orientation.
    """
    # use_flat_vehicle_coordinates=False gives us boxes in the sensor (Lidar) frame
    _, boxes, _ = nusc.get_sample_data(
        sample_data_token, use_flat_vehicle_coordinates=False
    )

    centers, half_sizes, rotations, labels, class_ids = [], [], [], [], []

    for box in boxes:
        centers.append(box.center)

        # FIX FOR ROTATION
        # NuScenes wlh = [width, length, height] (approx [y, x, z])
        # Rerun expects [x, y, z] extents.
        # We swap Width (0) and Length (1) to match the car's forward X-axis.
        w, l, h = box.wlh
        half_sizes.append([l / 2.0, w / 2.0, h / 2.0])

        # Quaternion: NuScenes [w, x, y, z] -> Rerun [x, y, z, w]
        q = box.orientation
        rotations.append([q[1], q[2], q[3], q[0]])

        labels.append(box.name)
        class_ids.append(abs(hash(box.name)) % 255)

    return centers, half_sizes, rotations, class_ids, labels


def get_boxes_2d(nusc, cam_token):
    """
    Get 3D boxes, project them to 2D camera plane, and return 2D rectangles.
    """
    # Get boxes in CAMERA coordinate frame
    _, boxes, cam_intrinsic = nusc.get_sample_data(
        cam_token, use_flat_vehicle_coordinates=False
    )

    boxes_2d_mins = []
    boxes_2d_sizes = []
    labels = []
    class_ids = []

    for box in boxes:
        # Check if box is visible (center must be in front of camera)
        # NuScenes 'box.center' is now relative to the camera
        if box.center[2] < 0:
            continue  # Skip boxes behind the camera

        # Project 3D corners to 2D pixel coordinates
        # box.corners() gives 3x8 matrix. view_points projects it.
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]

        # Compute axis-aligned bounding box from the projected corners
        min_x = np.min(corners_2d[0, :])
        max_x = np.max(corners_2d[0, :])
        min_y = np.min(corners_2d[1, :])
        max_y = np.max(corners_2d[1, :])

        # Filter boxes that are completely off-screen
        # (Assuming image size roughly 1600x900, but loose filtering is fine)
        if max_x < 0 or min_x > 1600 or max_y < 0 or min_y > 900:
            continue

        width = max_x - min_x
        height = max_y - min_y

        boxes_2d_mins.append([min_x, min_y])
        boxes_2d_sizes.append([width, height])
        labels.append(box.name)
        class_ids.append(abs(hash(box.name)) % 255)

    return boxes_2d_mins, boxes_2d_sizes, labels, class_ids


def log_cameras_and_boxes(nusc, sample):
    """Logs images and overlays 2D projected boxes."""
    cameras = [
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ]

    for cam_name in cameras:
        if cam_name in sample["data"]:
            cam_token = sample["data"][cam_name]
            cam_data = nusc.get("sample_data", cam_token)

            # A. Log Image
            img_path = os.path.join(nusc.dataroot, cam_data["filename"])
            if os.path.exists(img_path):
                img_bgr = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                rr.log(f"world/cameras/{cam_name}", rr.Image(img_rgb))

            # B. Log Projected 2D Boxes (on top of the image)
            mins, sizes, lbls, ids = get_boxes_2d(nusc, cam_token)
            if mins:
                rr.log(
                    f"world/cameras/{cam_name}/boxes",
                    rr.Boxes2D(mins=mins, sizes=sizes, labels=lbls, class_ids=ids),
                )


def visualize_fixed_scene(dataroot, scene_idx=0):
    if not os.path.exists(dataroot):
        print(f"Error: Path '{dataroot}' not found.")
        return

    print(f"Initializing NuScenes from {dataroot}...")
    version = "v1.0-mini" if "mini" in dataroot else "v1.0-trainval"
    try:
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    except:
        print("Failed to load NuScenes.")
        return

    rr.init("NuScenes 3D+2D Fixed", spawn=True)

    # Annotation Context (Colors)
    class_desc = []
    if hasattr(nusc, "lidarseg_idx2name_mapping"):
        for i, name in nusc.lidarseg_idx2name_mapping.items():
            class_desc.append((i, name, nusc.colormap[name]))
    if class_desc:
        rr.log("world", rr.AnnotationContext(class_desc), static=True)

    # Scene Loop
    if scene_idx >= len(nusc.scene):
        print("Scene index out of bounds.")
        return

    scene = nusc.scene[scene_idx]
    token = scene["first_sample_token"]
    frame_idx = 0

    print(f"Streaming Scene {scene_idx}: {scene['name']}...")

    while token != "":
        rr.set_time_sequence("frame_idx", frame_idx)

        sample = nusc.get("sample", token)
        lidar_token = sample["data"]["LIDAR_TOP"]

        # Log Cameras & 2D Boxes
        log_cameras_and_boxes(nusc, sample)

        # Log LiDAR Points (Green)
        lidar_data = nusc.get("sample_data", lidar_token)
        pcl_path = os.path.join(nusc.dataroot, lidar_data["filename"])
        pc = LidarPointCloud.from_file(pcl_path)
        points = pc.points.T[:, :3]

        rr.log("world/lidar", rr.Points3D(points, colors=[0, 255, 0], radii=0.08))

        # Log 3D Boxes (Corrected Rotation)
        c, h, r, ids, lbls = get_boxes_3d(nusc, lidar_token)
        if c:
            rr.log(
                "world/boxes",
                rr.Boxes3D(
                    centers=c, half_sizes=h, quaternions=r, class_ids=ids, labels=lbls
                ),
            )

        token = sample["next"]
        frame_idx += 1

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="NuScenes root path")
    parser.add_argument("--scene", type=int, default=0)
    args = parser.parse_args()

    visualize_fixed_scene(args.root, args.scene)
