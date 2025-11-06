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
from sklearn.decomposition import PCA
import ros2_numpy
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field


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
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    scale: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
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
    extr = config["calibration"]["extrinsics"]
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
    print("loading models")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=config["paths"]["yolo_model"],
        confidence_threshold=config["detection_params"]["confidence_threshold"],
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
        slice_height=config["sahi_params"]["slice_height"],
        slice_width=config["sahi_params"]["slice_width"],
        overlap_height_ratio=config["sahi_params"]["overlap_ratio"],
        overlap_width_ratio=config["sahi_params"]["overlap_ratio"],
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
    detections = detections.with_nms(
        threshold=config["detection_params"]["nms_threshold"]
    )
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

    # if len(detections) == 0:
    #     detections.mask = np.empty((0, *frame.shape[:2]), dtype=bool)
    #     return detections

    if len(detections) > 0:

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
                mask_2d.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            final_masks[idx] = resize_mask

        detections.mask = final_masks

    return detections


def compute_pca_bounding_box(point_cluster: np.ndarray, track_id: int) -> Optional[ObjectProperties3D]:
    if point_cluster.shape[0] < 3:
        print(f"  Track {track_id}: Not enough points ({point_cluster.shape[0]}) for PCA.")
        return None
    
    try:
        pca = PCA(n_components=3)
        pca.fit(point_cluster)
        center_3d = pca.mean_
        rotation_matrix = pca.components_.T
        if np.linalg.det(rotation_matrix) < 0:
            rotation_matrix[:, 2] *= -1
        transformed_points = pca.transform(point_cluster)
        min_vals = np.min(transformed_points, axis=0)
        max_vals = np.max(transformed_points, axis=0)
        scale_3d = max_vals - min_vals
        angle_z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        return ObjectProperties3D(
            center=center_3d,
            scale=scale_3d,
            angle_z=angle_z
        )
        
    except Exception as e:
        print(f" PCA failed for track {track_id}: {e}")
        return None

def compute_3d_object_properties(
    point_cluster: np.ndarray, track_id: int
) -> Optional[ObjectProperties3D]:
    if point_cluster.shape[0] < 4:
        return None  # not enough points

    cluster_pcd = o3d.geometry.PointCloud()
    cluster_pcd.points = o3d.utility.Vector3dVector(point_cluster)

    # RANSAC ground removal
    try:
        plane_model, inliers = cluster_pcd.segment_plane(
            distance_threshold=0.05, ransac_n=3, num_iterations=100
        )
        outlier_cloud = cluster_pcd.select_by_index(inliers, invert=True)
    except Exception as e:
        print(f"  RANSAC failed for track {track_id}: {e}")
        outlier_cloud = cluster_pcd  # Fallback

    # Calculate 3D Bounding Box
    if len(outlier_cloud.points) < 3:
        print(f"  Track {track_id}: Not enough points ({len(outlier_cloud.points)}) after RANSAC.")
        return None  # Not enough points after RANSAC

    return compute_pca_bounding_box(np.asarray(outlier_cloud.points), track_id)
    
    # check the box orientation code once (This for Qhull)
    # try:
    #     oriented_bbox_3d = outlier_cloud.get_oriented_bounding_box()
    #     return ObjectProperties3D(
    #         center=oriented_bbox_3d.center,
    #         scale=oriented_bbox_3d.extent,
    #         angle_z=np.arctan2(oriented_bbox_3d.R[1, 0], oriented_bbox_3d.R[0, 0]),
    #     )
    # except RuntimeError as e:
    #     print(f"  Qhull error for track {track_id}: {e}. Cluster is degenerate.")
    #     return None


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

        if distance_2d < config["tracking_params"]["movement_threshold_pixels"]:
            static_frames += 1
        else:
            static_frames = 0

        if static_frames >= config["tracking_params"]["static_frame_count_threshold"]:
            obj_status = "people.static"

        tracker_history[track_id] = (center_3d, center_2d, static_frames)
    else:
        tracker_history[track_id] = (center_3d, center_2d, 0)

    return obj_status, tracker_history


def format_yolo_txt_output(
    class_id: int, mask: np.ndarray, frame_shape: Tuple[int, int]
) -> Optional[str]:
    H, W = frame_shape
    if np.any(mask):
        polygons = sv.mask_to_polygons(mask)
        if polygons:
            segment = polygons[0] / np.array([W, H])
            segment_str = " ".join(map(str, segment.flatten()))
            return f"{class_id} {segment_str}"
    return None


def format_json_output(
    track_id: int, obj_status: str, props_3d: ObjectProperties3D
) -> Dict[str, Any]:
    return {
        "obj_id": str(track_id),
        "obj_type": obj_status,
        "psr": {
            "position": {
                "x": float(props_3d.center[0]),
                "y": float(props_3d.center[1]),
                "z": float(props_3d.center[2]),
            },
            "rotation": {"x": 0, "y": 0, "z": float(props_3d.angle_z)},
            "scale": {
                "x": float(props_3d.scale[0]),
                "y": float(props_3d.scale[1]),
                "z": float(props_3d.scale[2]),
            },
        },
    }


def process_frame_detection(
    frame_data: FrameData, tracker_history: Dict, config: Dict[str, Any]
) -> Tuple[List[Dict], List[str], List[Any], Dict]:

    json_frame_data = []
    yolo_text_lines = []
    valid_detections_for_viz = []
    custom_labels_for_viz = []
    frame_shape_hw = frame_data.frame.shape[:2]

    if frame_data.detections.mask is None:
        print(
            "Warning: Detections object has no mask attribute in process_frame_detections. Skipping frame."
        )
        return json_frame_data, yolo_text_lines, sv.Detections.empty(), tracker_history

    for idx in range(len(frame_data.detections)):
        # bbox_2d, mask, track_id, class_id = (
        #     detection.xyxy[0],
        #     detection.mask[0],
        #     detection.tracker_id[0],
        #     detection.class_id[0],
        # )
        bbox_2d = frame_data.detections.xyxy[idx]
        mask = frame_data.detections.mask[idx]
        track_id = frame_data.detections.tracker_id[idx]
        class_id = frame_data.detections.class_id[idx]
        detection = frame_data.detections[idx]

        # finding 3D point cluster
        u, v = frame_data.points_2d_image[:, 0], frame_data.points_2d_image[:, 1]
        mask_in_box = (
            (u >= bbox_2d[0]) & (u < bbox_2d[2]) & (v >= bbox_2d[1]) & (v < bbox_2d[3])
        )
        point_cluster = frame_data.points_3d_lidar[mask_in_box]

        # compute 3d properties
        props_3d = compute_3d_object_properties(point_cluster, track_id)
        if props_3d is None:
            props_3d = (
                ObjectProperties3D()
            )  # using the default as of now migth need to change in future

        # update state
        center_2d = get_center_point(bbox_2d)
        obj_status, tracker_history = update_tracking_state(
            track_id, props_3d, center_2d, tracker_history, config
        )

        json_obj = format_json_output(track_id, obj_status, props_3d)
        yolo_line = format_yolo_txt_output(class_id, mask, frame_shape_hw)

        json_frame_data.append(json_obj)
        if yolo_line:
            yolo_text_lines.append(yolo_line)

        valid_detections_for_viz.append(detection)
        custom_labels_for_viz.append(f"#{track_id} {obj_status.split('.')[1]}")

    # this might fail need to check
    if valid_detections_for_viz:
        viz_detections = sv.Detections.merge(valid_detections_for_viz)
        viz_detections.data["label"] = np.array(custom_labels_for_viz)
        # viz_detections.label = np.array(custom_labels_for_viz)
    else:
        viz_detections = sv.Detections.empty()

    return json_frame_data, yolo_text_lines, viz_detections, tracker_history


def save_outputs(
    frame: np.ndarray,
    frame_path: Path,
    json_data: List[Dict],
    yolo_text_lines: List[str],
    viz_detections: sv.Detections,
    tools: Tools,
    paths: Dict[str, Path],
):
    if json_data:
        json_fn = paths["labels_json"] / f"{frame_path.stem}.json"
        with open(json_fn, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

    if yolo_text_lines:
        label_fn = paths["labels_txt"] / f"{frame_path.stem}.txt"
        with open(label_fn, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_text_lines))

    annotated_frame = frame.copy()
    if len(viz_detections) > 0:
        if viz_detections.mask is not None:
            annotated_frame = tools.mask_annotater.annotate(
                annotated_frame, viz_detections
            )
        annotated_frame = tools.box_annotater.annotate(annotated_frame, viz_detections)
        annotated_frame = tools.label_annotater.annotate(
            annotated_frame, viz_detections
        )

    viz_fn = paths["visualizations"] / frame_path.name
    cv2.imwrite(str(viz_fn), annotated_frame)


def run_processing_pipeline(
    config: Dict[str, Any], data_dir: str, pcd_dir: str, output_dir: Optional[str]
):
    env = setup_environment(config, data_dir, pcd_dir, output_dir)
    models = load_models(config, env.device)
    tools = initialize_tools(config)

    tracker_history = {}
    image_paths = sorted(
        [
            p
            for p in env.paths["data"].glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )

    for i, image_path in enumerate(image_paths):
        vis_fn = env.paths["visualizations"] / image_path.name
        json_fn = env.paths["labels_json"] / f"{image_path.stem}.json"
        label_fn = env.paths["labels_txt"] / f"{image_path.stem}.txt"

        if vis_fn.exists() and json_fn.exists() and label_fn.exists():
            print(f"Skipping already processed frame: {image_path.name}")
            continue

        print(f"Processing frame {i+1}/{len(image_paths)}: {image_path.name}")

        try:
            frame_data_tuple = load_frame_data(image_path, env.paths["pcd"])
            # if frame_data_tuple is None:
            #     continue
            frame, points_3d_lidar = frame_data_tuple

            points_2d = project_lidar_to_image(
                points_3d_lidar, frame.shape[:2], env.lidar_to_cam_matrix
            )

            detections = run_2d_pipeline(frame, models, tools, config)
            # if len(detections) == 0:
            #     print(
            #         f"  No objects tracked in {image_path.name}. Saving original frame."
            #     )
            #     cv2.imwrite(str(vis_fn), frame)  # Save original frame
            #     continue
            detections = run_segmentation(
                frame, detections, models.sam_predictor, env.device
            )

            frame_data = FrameData(
                frame=frame,
                points_3d_lidar=points_3d_lidar,
                points_2d_image=points_2d,
                detections=detections,
            )

            # Process Detections (3D BBox, State, Formatting)
            json_data, yolo_lines, viz_detections, tracker_history = (
                process_frame_detection(frame_data, tracker_history, config)
            )

            save_outputs(
                frame,
                image_path,
                json_data,
                yolo_lines,
                viz_detections,
                tools,
                env.paths,
            )
            print(f"Frame {image_path.name} processed. Results saved.")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory on frame {image_path.name}, skipping")
                torch.cuda.empty_cache()
                time.sleep(10)
                continue
            elif "qhull input error" in str(e):
                print(
                    f"Qhull error (degenerate cluster) on frame {image_path.name}, skipping 3D properties."
                )
                continue
            else:
                print(f"RuntimeError on {image_path.name}: {e}")
                continue

        # except Exception as e:
        #     print(f"Unhandled exception processing {image_path.name}: {e}")
        #     continue

    print(f"Full sequence processing complete. Results saved in: {env.paths['output']}")


def main():
    """
    Parses arguments, loads config, and starts the processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Advanced annotation and tracking with YOLO, SAHI, SAM, ByteTrack"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the directory containing input image frames.",
    )
    parser.add_argument(
        "--pcd_dir",
        type=str,
        required=True,
        help="Path to the directory of corresponding .pcd files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the directory where all results will be saved. (Optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        # default="config.yml",
        help="Path to the configuration YAML file. (Optional)",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

            if config is None:
                print(f"Error: Config file at {args.config} is empty or malformed.")
                exit()

    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()

    run_processing_pipeline(
        config=config,
        data_dir=args.data,
        pcd_dir=args.pcd_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
