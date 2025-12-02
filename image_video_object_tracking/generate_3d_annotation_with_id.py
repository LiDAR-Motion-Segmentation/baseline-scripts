import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import cv2
import open3d as o3d
from pandas.core.indexers import validate_indices
from scipy.spatial.transform import Rotation as R
import json
from sklearn.cluster import DBSCAN
from torch._C import P
from tqdm import tqdm
from collections import Counter

FIXED_IMG_SHAPE = (720, 1280)


class CameraConfig:
    def __init__(
        self, name: str, intr_path: Path, extr_path: Path, json_dir: Path
    ) -> None:
        self.name = name
        self.intr = np.loadtxt(str(intr_path)).reshape(3, 3)
        self.extr = self._load_extrinsics(extr_path)
        self.json_dir = json_dir

    def _load_extrinsics(self, p: Path) -> np.ndarray:
        # Expected format: x y z qx qy qz qw
        vals = np.loadtxt(str(p))
        if vals.size != 7:
            raise ValueError(
                f"Extrinsics file {p} must have 7 values (x, y, z, qx, qy, qz, qw)"
            )
        x, y, z, qx, qy, qz, qw = vals
        m = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = m
        T[:3, 3] = [x, y, z]
        return T


def project_lidar_to_image(xyz: np.ndarray, extr: np.ndarray, intr: np.ndarray):
    """Projects 3D LiDAR points to 2D Image plane.
    Returns:
        uv: 2D pixel coordinates
        valid: Boolean mask of points in front of the camera
    """
    # Note: Usually extrinsics are Lidar-to-Cam. If Cam-to-Lidar, invert T.
    # Assuming T is Lidar -> Camera based on standard datasets.
    T_inv = np.linalg.inv(extr)
    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])  # Homogenous
    pts_cam = (T_inv @ pts_h.T).T[:, :3]
    valid_z = pts_cam[:, 2] > 0

    # Project to Image Plane (Pinhole)
    # uv = K * [x/z, y/z, 1]
    pts_cam_valid = pts_cam[valid_z]
    if len(pts_cam_valid) == 0:
        return np.zeros((0, 2)), valid_z

    proj = (intr @ pts_cam_valid.T).T
    uv_valid = proj[:, :2] / proj[:, 2:3]

    # mapping to orginal size
    uv = np.zeros((xyz.shape[0], 2), dtype=np.float32)
    uv[valid_z] = uv_valid
    return uv, valid_z


def get_points_with_ids(
    xyz: np.ndarray,
    uv: np.ndarray,
    valid_mask: np.ndarray,
    polygons: List[List[float]],
    ids: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        points: (N, 3) array of 3D points
        point_ids: (N,) array of IDs corresponding to each point
        point_indices
    """
    h, w = FIXED_IMG_SHAPE

    # We use an integer mask to store IDs directly on the image plane
    # -1 = Background, >=0 = Global ID
    id_mask = np.full((h, w), -1, dtype=np.int32)

    # draw polygon with their id
    for poly, gid in zip(polygons, ids):
        pts = np.array(poly).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(id_mask, [pts], int(gid))

    valid_indices = np.where(valid_mask)[0]
    uv_valid = np.round(uv[valid_mask]).astype(int)
    in_bounds = (
        (uv_valid[:, 0] >= 0)
        & (uv_valid[:, 0] < w)
        & (uv_valid[:, 1] >= 0)
        & (uv_valid[:, 1] < h)
    )

    valid_indices = valid_indices[in_bounds]
    uv_clamped = uv_valid[in_bounds]

    # sample the ID mask
    point_ids = id_mask[uv_valid[:, 1], uv_clamped[:, 0]]

    # Filter only points that hit a person (ID != -1)
    is_person = point_ids != 1

    final_indices = valid_indices[is_person]
    final_ids = point_ids[is_person]

    return xyz[final_indices], final_ids, final_indices


def cluster_and_box_with_id(
    points: np.ndarray, point_ids: np.ndarray, obj_type: str
) -> List[dict]:
    # Cluster with ID Voting

    clustering = DBSCAN(eps=0.5, min_samples=10).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    annotations = []

    for k in unique_labels:
        if k == -1:
            continue

        # getting point in the cluster
        mask = labels == k
        cluster_pts = points[mask]
        cluster_ids = point_ids[mask]

        # voting logic to find the most frequent ID in this cluster
        if len(cluster_ids) == 0:
            continue
        most_common_id = Counter(cluster_ids).most_common(1)[0][0]

        # fit OBB
        pcd = o3d.geometry.PointClouds()
        pcd.points = o3d.utility.Vector3dVector(cluster_pts)

        try:
            obb = pcd.get_oriented_bounding_box()
            center = obb.center
            extent = obb.extent
            R_mat = obb.R
            yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
            ann = {
                "obj_id": str(most_common_id),
                "obj_type": obj_type,
                "psr": {
                    "position": {
                        "x": float(center[0]),
                        "y": float(center[1]),
                        "z": float(center[2]),
                    },
                    "rotation": {"x": 0.0, "y": 0.0, "z": float(yaw)},
                    "scale": {
                        "x": float(extent[0]),
                        "y": float(extent[1]),
                        "z": float(extent[2]),
                    },
                },
            }
            annotations.append(ann)

        except Exception as e:
            print(f"Warning: Failed to fit box for cluster {k}: {e}")

    return annotations


def process_single_frame(
    pcd_path: Path, cameras: List[CameraConfig], output_dir: Path, pcd_out_dir: Path
):
    frame_name = pcd_path.stem
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    xyz_all = np.asarray(pcd.points)

    # Lists for Annotation Logic
    mov_pts_list, mov_ids_list = [], []
    stat_pts_list, stat_ids_list = [], []

    # Lists for Coloring Logic (Just indices)
    all_moving_indices = []
    all_static_indices = []

    IMG_SHAPE = (720, 1280)

    for cam in cameras:
        json_file = cam.json_dir / f"{frame_name}.json"
        if not json_file.exists():
            continue
        with open(json_file, "r") as f:
            detections = json.load(f)
        polys_mov, ids_mov = [], []
        polys_stat, ids_stat = [], []

        for obj in detections:
            poly = obj.get("segmentation_polygon", [])
            status = obj.get("status", "Unknown")
            gid = obj.get("global_id", -1)

            if not poly or gid == -1:
                continue

            if status == "Moving":
                polys_mov.append(poly)
                ids_mov.append(gid)
            elif status == "Static":
                polys_stat.append(poly)
                ids_stat.append(gid)

        uv, valid = project_lidar_to_image(xyz_all, cam.extr, cam.intr)

        if polys_mov:
            pts, ids, idxs = get_points_with_ids(xyz_all, uv, valid, polys_mov, ids_mov)
            mov_pts_list.append(pts)
            mov_ids_list.append(ids)
            all_moving_indices.extend(idxs.tolist())

        if polys_stat:
            pts, ids, idxs = get_points_with_ids(
                xyz_all, uv, valid, polys_stat, ids_stat
            )
            stat_pts_list.append(pts)
            stat_ids_list.append(ids)
            all_static_indices.extend(idxs.tolist())

    final_mov_pts = np.vstack(mov_pts_list) if mov_pts_list else np.empty((0, 3))
    final_mov_ids = (
        np.concatenate(mov_ids_list) if mov_ids_list else np.empty((0,), dtype=int)
    )

    final_stat_pts = np.vstack(stat_pts_list) if stat_pts_list else np.empty((0, 3))
    final_stat_ids = (
        np.concatenate(stat_ids_list) if stat_ids_list else np.empty((0,), dtype=int)
    )

    annotations = []
    annotations.extend(
        cluster_and_box_with_id(final_mov_pts, final_mov_ids, "moving_people")
    )
    annotations.extend(
        cluster_and_box_with_id(final_stat_pts, final_stat_ids, "static_people")
    )

    with open(output_dir / f"{frame_name}.json", "w") as f:
        json.dump(annotations, f, indent=2)

    # Default Color: Green [0, 1, 0] for Environment
    colors = np.zeros((xyz_all.shape[0], 3))
    colors[:, 1] = 1.0  # Set G=1.0

    # Color Moving: Red [1, 0, 0]
    if all_moving_indices:
        # Use unique indices to handle overlapping camera views
        unique_mov = list(set(all_moving_indices))
        colors[unique_mov] = [1.0, 0.0, 0.0]

    # Color Static: Blue [0, 0, 1]
    if all_static_indices:
        unique_stat = list(set(all_static_indices))
        colors[unique_stat] = [0.0, 0.0, 1.0]

    # Save PCD
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(xyz_all)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(str(pcd_out_dir / f"{frame_name}.pcd"), pcd_colored)


def main():
    parser = argparse.ArgumentParser()
    # ARG FORMAT: <NAME> <INTR_FILE> <EXTR_FILE> <JSON_DIR> (Only 4 args now)
    parser.add_argument("--cam", nargs=4, action="append", required=True)
    parser.add_argument("--pcd_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    cameras = [CameraConfig(c[0], Path(c[1]), Path(c[2]), Path(c[3])) for c in args.cam]
    pcd_dir = Path(args.pcd_dir)

    # Setup Output Dirs
    out_json_dir = Path(args.out_dir) / "json_labels"
    out_pcd_dir = Path(args.out_dir) / "colored_pcd"
    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_pcd_dir.mkdir(parents=True, exist_ok=True)

    pcd_files = sorted(list(pcd_dir.glob("*.pcd")))
    print(f"Processing {len(pcd_files)} frames")

    for pcd in tqdm(pcd_files):
        try:
            process_single_frame(pcd, cameras, out_json_dir, out_pcd_dir)
        except Exception as e:
            print(f"Err {pcd.name}: {e}")


if __name__ == "__main__":
    main()
