import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
from sklearn import DBSCAN


class CameraConfig:
    def __init__(
        self, name: str, img_dir: Path, intr_path: Path, extr_path: Path, lbl_dir: Path
    ):
        self.name = name
        self.img_dir = img_dir
        self.intr = np.loadtxt(str(intr_path)).reshape(3, 3)
        self.extr = self._load_extrinsics(extr_path)
        self.lbl_dir = lbl_dir

    def _load_extrinsics(self, p: Path) -> np.ndarray:
        vals = np.loadtxt(str(p))
        x, y, z, qx, qy, qz, qw = vals

        # might crash need to check
        m = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = m
        T[:3, 3] = [x, y, z]
        return T


def build_mask_from_label(label_file: Path, shape: tuple) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if not label_file.exists():
        return mask
    with open(label_file, "r") as f:
        for line in f:
            arr = np.array([float(x) for x in line.strip().split()])
            if len(arr) < 7:
                continue
            pts = (arr[1:].reshape(-1, 2) * [w, h]).astype(np.int32)
            if pts.shape[0] > 2:
                cv2.fillPoly(mask, [pts], 1)
    return mask


def project_lidar_to_image(xyz: np.ndarray, extr: np.ndarray, intr: np.ndarray):
    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    T_inv = np.linalg.inv(extr)
    pts_cam = (T_inv @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > 0
    pts_cam_valid = pts_cam[valid]
    proj = (intr @ pts_cam_valid.T).T
    uv = np.zeros((xyz.shape[0], 2), dtype=np.float32)
    uv_valid = proj[:, :2] / proj[:, 2:3]
    uv[valid] = uv_valid
    return uv, valid


def mask_point_indices(
    xyz: np.ndarray, uv: np.ndarray, valid: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    h, w = mask.shape
    pixel_uv = np.round(uv[valid]).astype(int)
    in_img = (
        (pixel_uv[:, 0] >= 0)
        & (pixel_uv[:, 0] < w)
        & (pixel_uv[:, 1] >= 0)
        & (pixel_uv[:, 1] < h)
    )
    pixels_uv_in = pixel_uv[in_img]
    mask_vals = mask[pixels_uv_in[:, 1], pixels_uv_in[:, 0]]
    final_mask = np.where(valid)[0][in_img][mask_vals == 1]
    return final_mask


def get_clusters_masked_points(
    points: np.ndarray, eps: float = 0.5, min_samples: int = 8
) -> List[np.ndarray]:
    if len(points) == 0:
        return []
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(points)
    cluster = []
    for k in set(labels):
        if k == -1:
            continue
        cluster.append(points[labels == k])
    return cluster


def save_3d_annotations_json(
    clusters: List[np.ndarray], out_json: Path, obj_type: str = "moving_people"
) -> None:
    annotations = []
    for i, pts in enumerate(clusters, 1):
        if len(pts) < 3:
            continue
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        obb = pc.get_oriented_bounding_box()
        ann = {
            "obj_id": str(i),
            "obj_type": obj_type,
            "psr": {
                "position": {
                    "x": float(obb.center[0]),
                    "y": float(obb.center[1]),
                    "z": float(obb.center[2]),
                },
                "rotation": {
                    "x": 0,
                    "y": 0,
                    "z": float(np.arctan2(obb.R[1, 0], obb.R[0, 0])),
                },
                "scale": {
                    "x": float(obb.extent[0]),
                    "y": float(obb.extent[1]),
                    "z": float(obb.extent[2]),
                },
            },
        }
        annotations.append(ann)
    with open(out_json, "w") as f:
        json.dump(annotations, f, indent=2)
