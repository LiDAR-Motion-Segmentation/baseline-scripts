import argparse
import cv2
import numpy as np
import open3d as o3d
import ros2_numpy
import yaml
from pathlib import Path


def process_frame(
    pcd_path: Path,
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    transform_matrix: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    print(f"Procesing: {pcd_path.name}")
    pc_data = o3d.io.read_point_cloud(str(pcd_path))
    xyz_points = np.asarray(pc_data.points)  # Original 3D points in LiDAR space
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Warning: Could not load image at {image_path}. Skipping frame.")
        return

    all_polygons = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                poly_data = np.array([float(x) for x in line.split()])
                all_polygons.append(poly_data)
    except FileNotFoundError:
        print(f"Warning: Label file not found at {label_path}. Skipping frame.")
        return

    # 2D segmentation mask from labels
    h, w, _ = image.shape
    segmentation_mask = np.zeros((h, w), dtype=np.uint8)
    overlay = image.copy()
    for poly_data in all_polygons:
        if len(poly_data) >= 7:
            try:
                normalized_points = poly_data[1:].reshape(-1, 2)
                pixel_points = (normalized_points * np.array([w, h])).astype(np.int32)

                if pixel_points.shape[0] >= 3:
                    cv2.fillPoly(overlay, [pixel_points], color=(0, 0, 255))
                    cv2.fillPoly(segmentation_mask, [pixel_points], color=1)
            except ValueError:
                print(f"Warning: Skipping malformed polygon in {label_path.name}.")

    alpha = 0.4
    final_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    cv2.imwrite(str(output_dir / "sam_image" / f"{pcd_path.stem}.png"), final_image)

    points_homogeneous = np.hstack([xyz_points, np.ones((xyz_points.shape[0], 1))])
    points_camera_frame = (transform_matrix @ points_homogeneous.T).T[:, :3]

    # filter out points that are behind the camera
    z_forward_mask = points_camera_frame[:, 2] > 0
    points_3d_forward = points_camera_frame[z_forward_mask]
    orginal_indices = np.arange(len(xyz_points))
    indices_forward = orginal_indices[z_forward_mask]

    points_2d = None
    if len(points_3d_forward) > 0:
        points_2d, _ = cv2.projectPoints(
            points_3d_forward, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
        )
        points_2d = points_2d.squeeze(axis=1)  # (N,1,2) to (N,2)

    if points_2d is None or len(points_2d) == 0:
        print(f"warning: no valid 3D points to project for {pcd_path.name}")
        return

    # Find points that are within the image boundaries, check once
    on_image_mask = (
        (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < w)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < h)  # This will now correctly filter out 1112
    )

    valid_pixels = points_2d[on_image_mask].astype(int)
    indices_on_image = indices_forward[on_image_mask]

    # if len(valid_pixels) == 0:
    #     print(f"Warning: No LiDAR points projected onto the image for {pcd_path.name}.")
    #     return

    if len(valid_pixels) > 0:
        # Look up the mask value for each valid pixel
        mask_values = segmentation_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
        person_mask_on_image = mask_values == 1

        # Final Filter: Get the original indices of points that pass all 3 filters
        person_indices = indices_on_image[person_mask_on_image]
    else:
        print(f"Warning: No LiDAR points projected onto the image for {pcd_path.name}.")

    person_points = xyz_points[person_indices]
    np.save(
        str(output_dir / "sam_lidar_segmented" / f"{pcd_path.stem}.npy"), person_points
    )

    length_xyz_points = len(xyz_points)
    colors = np.full((length_xyz_points, 3), [0.0, 1.0, 0.0])  # green for non person
    colors[person_indices] = [1.0, 0.0, 0.0]  # red for person
    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(xyz_points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(
        str(output_dir / "sam_lidar" / f"{pcd_path.stem}.pcd"), colored_pcd
    )


def main():
    parser = argparse.ArgumentParser(
        description="Project LiDAR points onto segmented images."
    )
    parser.add_argument(
        "--pcd_dir",
        type=str,
        required=True,
        help="Path to the directory of input .pcd files.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Path to the directory of input .png images.",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        required=True,
        help="Path to the directory of input .txt label files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the base directory for saving results.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=3,
        help="Frame offset between PCD and image/label files.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    T_lidar_camera_arr = config["T_lidar_camera"]
    translation_m = ros2_numpy.geometry.transformations.translation_matrix(
        T_lidar_camera_arr[0:3]
    )
    rotation_m = ros2_numpy.geometry.transformations.quaternion_matrix(
        T_lidar_camera_arr[3:7]
    )
    T_lidar_camera = np.dot(translation_m, rotation_m)
    transform_matrix = np.linalg.inv(T_lidar_camera)

    intr = config["intrinsics"]
    camera_matrix = np.array(
        [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]
    )
    dist_coeffs = np.array(intr["distortion"])

    output_dir = Path(args.output_dir)
    (output_dir / "sam_lidar").mkdir(parents=True, exist_ok=True)
    (output_dir / "sam_image").mkdir(parents=True, exist_ok=True)
    (output_dir / "sam_lidar_segmented").mkdir(parents=True, exist_ok=True)

    pcd_files = sorted(Path(args.pcd_dir).glob("*.pcd"))
    for pcd_path in pcd_files:
        try:
            timestep = int(pcd_path.stem)
            image_timestep = timestep + args.offset

            image_path = Path(args.image_dir) / f"{image_timestep:06d}.png"
            label_path = Path(args.label_dir) / f"{image_timestep:06d}.txt"

            process_frame(
                pcd_path,
                image_path,
                label_path,
                output_dir,
                transform_matrix,
                camera_matrix,
                dist_coeffs,
            )
        except Exception as e:
            print(f"Error processing {pcd_path.name}: {e}")


if __name__ == "__main__":
    main()
