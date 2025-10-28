import os
import cv2
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
import argparse


def load_calibration_yaml(calib_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(calib_path, "r") as f:
        calib = yaml.safe_load(f)
    trans = np.array(calib["extrinsics"]["translation"])
    quat = np.array(calib["extrinsics"]["rotation"])
    qx, qy, qz, qw = quat
    R = np.array(
        [
            [
                1 - 2 * qy**2 - 2 * qz**2,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx**2 - 2 * qz**2,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx**2 - 2 * qy**2,
            ],
        ]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = trans
    return T, calib


def project_lidar_to_equirect(
    xyz_points, T_lidar_to_cam, intrinsics
) -> tuple[np.ndarray, np.ndarray]:
    points_hom = np.hstack([xyz_points, np.ones((xyz_points.shape[0], 1))])
    points_cam = (T_lidar_to_cam @ points_hom.T).T[:, :3]
    h = intrinsics["height"]
    w = intrinsics["width"]
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-7
    theta = np.arctan2(x, z)
    phi = np.arcsin(y / r)
    u = ((theta + np.pi) / (2 * np.pi)) * w
    v = ((np.pi / 2 - phi) / np.pi) * h
    pixel_coords = np.stack([u, v], axis=1)
    valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    return pixel_coords, valid_mask


def polygon_to_mask(polygons, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    for poly in polygons:
        pts = (poly * np.array(image_shape[::-1])).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def visualize_polygons(image, polygons, mask, out_path):
    overlay = image.copy()
    for poly in polygons:
        pts = (poly * np.array(image.shape[:2][::-1])).astype(np.int32)
        cv2.polylines(overlay, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.fillPoly(overlay, [pts], color=(0, 0, 255))
    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.imwrite(out_path, blended)


def process(
    image_path, pcd_path, label_path, calib_path, img_output_dir, pcd_output_dir
):
    image = cv2.imread(str(image_path))
    pc = o3d.io.read_point_cloud(str(pcd_path))
    xyz = np.asarray(pc.points)
    T_lidar_to_camera, calib = load_calibration_yaml(calib_path)
    intrinsics = calib["intrinsics"]

    polygons = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip().split()
            if not line:
                continue
            arr = np.array([float(r) for r in line[1:]])
            polygons.append(arr.reshape(-1, 2))
    mask = polygon_to_mask(polygons, image.shape[:2])

    Path(img_output_dir).mkdir(exist_ok=True)
    vis_path = os.path.join(img_output_dir, f"{Path(image_path).stem}_sam_overlay.png")
    visualize_polygons(image, polygons, mask, vis_path)

    pixel_coords, valid_mask = project_lidar_to_equirect(
        xyz, T_lidar_to_camera, intrinsics
    )
    pixel_coords_int = pixel_coords[valid_mask].astype(int)
    mask_values = mask[pixel_coords_int[:, 1], pixel_coords_int[:, 0]]
    inside_mask = mask_values == 1

    points_masked_indices = np.where(valid_mask)[0][inside_mask]
    points_masked = xyz[points_masked_indices]
    # out_masked_fn = os.path.join(output_dir, f"{Path(image_path).stem}_sam_lidar_segmented.npy")
    # np.save(out_masked_fn, points_masked)

    colors = np.zeros_like(xyz)
    colors[:] = [0, 1, 0]
    colors[points_masked_indices] = [1, 0, 0]
    color_pcd = o3d.geometry.PointCloud()
    color_pcd.points = o3d.utility.Vector3dVector(xyz)
    color_pcd.colors = o3d.utility.Vector3dVector(colors)
    Path(pcd_output_dir).mkdir(exist_ok=True)
    o3d.io.write_point_cloud(
        os.path.join(pcd_output_dir, f"{Path(image_path).stem}.pcd"), color_pcd
    )
    print(f"Completed {image_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SAM mask back-projection with calibration"
    )
    parser.add_argument("--images", required=True, help="Directory with images")
    parser.add_argument("--pcds", required=True, help="Directory with PCD files")
    parser.add_argument(
        "--labels", required=True, help="Directory with polygon labels (.txt)"
    )
    parser.add_argument("--calib", required=True, help="Calibration YAML file")
    parser.add_argument(
        "--img_output", required=True, help="Output directory for images"
    )
    parser.add_argument("--pcd_output", required=True, help="Output directory for PCD")
    args = parser.parse_args()

    image_files = sorted(list(Path(args.images).glob("*.png")))
    pcd_files = sorted(list(Path(args.pcds).glob("*.pcd")))
    label_files = sorted(list(Path(args.labels).glob("*.txt")))
    assert len(image_files) == len(pcd_files) == len(label_files)
    for img, pcd, lbl in zip(image_files, pcd_files, label_files):
        process(img, pcd, lbl, args.calib, args.img_output, args.pcd_output)


if __name__ == "__main__":
    main()
