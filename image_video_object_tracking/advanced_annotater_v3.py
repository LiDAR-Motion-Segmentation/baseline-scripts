from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import supervision as sv
import math
import yaml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from segment_anything_hq import SamPredictor, sam_model_registry
import argparse
import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import ros2_numpy
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class EnvironmentSetting:
    device: torch.device
    paths: Dict[str, Path]
    lidar_to_cam_matrix: np.ndarray


@dataclass
class Models:
    detection_model: AutoDetectionModel
    sam_predictor: SamPredictor


@dataclass
class Tools:
    tracker: sv.ByteTrack
    box_annotater: sv.BoxAnnotator
    mask_annotater: sv.MaskAnnotator
    label_annotater: sv.LabelAnnotator


@dataclass
class FrameData:
    frame: np.ndarray
    points_3d_lidar: np.ndarray
    points_2d_image: np.ndarray
    detections: sv.Detections


@dataclass
class ObjectProperties3D:
    center: np.ndarray = np.ndarray([0, 0, 0])
    scale: np.ndarray = np.ndarray([0, 0, 0])
    angle_z: float = 0.0


@dataclass
class ObjectData:
    json_obj: Dict[str, Any]
    label: str
    yolo_txt_line: Optional[str]


def get_center_point(bbox: np.ndarray) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def setup_environment(
    config: Dict[str, Any], data_dir: str, pcd_dir: str, output_dir: Optional[str]
) -> EnvironmentSetting:

    print("Executing setting up environment code")

    device = torch.device(
        config["models"]["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    data_path = Path(data_dir)
    if not output_dir:
        output_dir_path = data_path.parent / f"{data_path.stem}_advanced_annotations"
    else:
        output_dir_path = Path(output_dir)

    paths = {
        "data": data_path,
        "pcd": Path(pcd_dir),
        "output": output_dir_path,
        "labels_txt": output_dir_path / "labels_txt",
        "labels_json": output_dir_path / "labels_json",
        "visualizations": output_dir_path / "visualizations",
    }
    for path in paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)

    # ros2_numpy version
    extr = config["calib"]["extrinsics"]
    t = extr["translation"]
    q = extr["rotation"]
    translation_m = ros2_numpy.geometry.transformations.translation_matrix(t)
    rotation_m = ros2_numpy.geometry.transformations.quaternion_matrix(q)
    T_lidar_camera = np.dot(translation_m, rotation_m)
    lidar_to_cam_matrix = np.linalg.inv(T_lidar_camera)

    return EnvironmentSetting(
        device=device, paths=paths, lidar_to_cam_matrix=lidar_to_cam_matrix
    )


def load_models(config: Dict[str, Any], device: torch.device) -> Models:
    print("lodaing models")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=config["paths"]["yolo_model"],
        confidence_threshold=config["detection"]["confidence_threshold"],
        device=device,
    )

    sam = sam_model_registry[config["models"]["sam_model_type"]]()
    state_dict = torch.load(config["paths"]["sam_checkpoint"], map_location="cpu")
    sam.load_state_dict(state_dict)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    return Models(detection_model=detection_model, sam_predictor=sam_predictor)


def initialize_tools(config: Dict[str, Any]) -> Tools:
    print("initialized tools")

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)

    label_annotater = sv.LabelAnnotator(text_scale=0.6, text_color=sv.Color.BLACK)

    return Tools(
        tracker=tracker,
        box_annotater=box_annotator,
        mask_annotater=mask_annotator,
        label_annotater=label_annotater,
    )


def load_frame_data(
    image_path: Path, pcd_base_path: Path
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    frame = cv2.imread(str(image_path))
    pcd_file = pcd_base_path / f"{image_path.stem}.pcd"

    if not pcd_file.exists():
        print(f"Warning: Point cloud not found for {image_path.name}. Skipping.")
        return None

    pcd = o3d.io.read_point_cloud(str(pcd_file))
    points_3d_lidar = np.asarray(pcd.points)

    if frame is None:
        print(f"Warning: Could not read image {image_path.name}. Skipping.")
        return None

    return frame, points_3d_lidar


def run_2d_pipeline(
    frame: np.ndarray, models: Models, tools: Tools, config: Dict[str, Any]
) -> sv.Detections:
    sahi_result = get_sliced_prediction(
        frame,
        models.detection_model,
        slice_height=config["sahi"]["slice_height"],
        slice_width=config["sahi"]["slice_width"],
        overlap_height_ratio=config["sahi"]["overlap_ratio"],
        overlap_width_ratio=config["sahi"]["overlap_ratio"],
    )

    xyxy_list, confidence_list, class_id_list = [], [], []
    if sahi_result.object_prediction_list:
        for pred in sahi_result.object_prediction_list:
            xyxy_list.append(pred.bbox.to_xyxy())
            confidence_list.append(pred.score.value)
            class_id_list.append(pred.category.id)

    detections = sv.Detections(
        xyxy=np.array(xyxy_list),  # size (0,4)
        confidence=np.array(confidence_list),  # size (0)
        class_id=np.array(class_id_list).astype(int),  # size (0)
    )

    detections = detections[detections.class_id == 0]  # filtering people
    detections = detections.with_nms(threshold=config["detection"]["nms_theshold"])
    tracked_detection = tools.tracker.update_with_detections(detections)
    tracked_detection = tracked_detection[tracked_detection.tracker_id != None]

    return tracked_detection


def project_lidar_to_image(
    points_3d_lidar: np.ndarray,
    frame_shape: Tuple[int, int],
    lidar_to_cam_matrix: np.ndarray,
) -> np.ndarray:
    H, W = frame_shape

    # lidar points -> 3d cam space using spherical equipolar projection
    points_homogeneous = np.hstack(
        [points_3d_lidar, np.ones((points_3d_lidar.shape[0], 1))]
    )
    points_3d_cam = (lidar_to_cam_matrix @ points_homogeneous.T).T[:, :3]
    x, y, z = points_3d_cam[:, 0], points_3d_cam[:, 1], points_3d_cam[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    points_2d = np.full((x.shape[0], 2), -1.0)
    valid_indices = r > 0.001

    x_valid, y_valid, z_valid, r_valid = (
        x[valid_indices],
        y[valid_indices],
        z[valid_indices],
        r[valid_indices],
    )
    p_x, p_y, p_z = x_valid / r_valid, y_valid / r_valid, z_valid / r_valid

    p_y = np.clip(p_y, -1.0, 1.0)
    phi = np.arcsin(p_y)
    theta = np.arctan2(p_x, p_z)

    u_coords = (theta * W / (2 * np.pi)) + (W / 2)
    v_coords = (phi * H / np.pi) + (H / 2)

    points_2d[valid_indices] = np.vstack((u_coords, v_coords)).T
    return points_2d


def run_segmentation(
    frame: np.ndarray,
    detections: sv.Detections,
    sam_predictor: SamPredictor,
    device: torch.device,
) -> sv.Detections:
    if len(detections) == 0:
        return detections

    # Resize frame and boxes for SAM
    H, W, _ = frame.shape
    scale = 1024 / max(H, W)

    frame_for_sam = cv2.resize(frame, (int(W * scale), int(H * scale)))
    sam_predictor.set_image(frame_for_sam)
    scaled_boxes = detections.xyxy * scale

    # mask
    masks_tensor, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=torch.tensor(scaled_boxes).to(device),
        multimask_output=False,
    )
    masks_np = masks_tensor.cpu().numpy()  # shape (N, 1, H_scaled, W_scaled)

    # resizing the mask to the orignal image
    num_detections = len(detections)
    final_masks = np.zeros((num_detections, H, W), dtype=bool)
    for idx, mask in enumerate(masks_np):
        mask_2d = mask.squeeze()  # (H_scaled, W_scaled)
        resize_mask = cv2.resize(
            mask_2d.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        final_masks[idx] = resize_mask

    detections.mask = final_masks
    return detections


def compute_3d_object_properties(
    point_cluster: np.ndarray, track_id: int
) -> Optional[ObjectProperties3D]:
    if point_cluster.shape[0] < 4:
        return None  # not enough points

    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(point_cluster)

    # RANSAC ground removal
    try:
        plane_model, inliers = cluster_pcd.segement_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=100
        )
        outlier_cloud = cluster_pcd.select_by_index(inliers, invert=True)
    except Exception as e:
        print(f"  RANSAC failed for track {track_id}: {e}")
        outlier_cloud = cluster_pcd  # Fallback

    # Calculate 3D Bounding Box
    if len(outlier_cloud.points) < 4:
        return None  # Not enough points after RANSAC

    # check the box orientation code once
    try:
        oriented_bbox_3d = outlier_cloud.get_oriented_bounding_box()
        return ObjectProperties3D(
            center=oriented_bbox_3d.center,
            scale=oriented_bbox_3d.extent,
            angle_z=np.arctan2(oriented_bbox_3d.R[1, 0], oriented_bbox_3d.R[0, 0]),
        )
    except RuntimeError as e:
        print(f"  Qhull error for track {track_id}: {e}. Cluster is degenerate.")
        return None


def update_tracking_state(
    track_id: int,
    props_3d: ObjectProperties3D,
    center_2d: Tuple[float, float],
    tracker_history: Dict,
    config: Dict[str, Any],
) -> Tuple[str, Dict]:
    obj_status = "people.moving"
    static_frames = 0
    center_3d = props_3d.center

    if track_id in tracker_history:
        last_center_3d, last_center_2d, static_frames = tracker_history[track_id]
        distance_2d = math.dist(center_2d, last_center_2d)

        if distance_2d < config["tracking"]["movement_threshold_pixels"]:
            static_frames += 1
        else:
            static_frames = 0

        if static_frames >= config["tracking"]["static_frame_count_threshold"]:
            obj_status = "people.static"

        tracker_history[track_id] = (center_3d, center_2d, static_frames)
    else:
        tracker_history[track_id] = (center_3d, center_2d, 0)

    return obj_status, tracker_history
