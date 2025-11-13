import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json


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
