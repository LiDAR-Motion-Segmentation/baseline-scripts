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


def get_center_point(bbox: np.ndarray) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def get_transform_matrix(translation: list, rotation_quat: list) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = R.from_quat(rotation_quat).as_matrix()
    transform[:3, 3] = translation
    return transform


def track_and_annotate(
    data: str | Path,
    pcd_dir: str | Path,
    config: dict,
    output_dir: str | Path | None = None,
):
    print("initializing models")
    data_path = Path(data)
    pcd_path = Path(pcd_dir)

    cfg = {
        "paths": config["paths"],
        "models": config["models"],
        "sahi": config["sahi_params"],
        "detection": config["detection_params"],
        "tracking": config["tracking_params"],
        "calib": config["calibration"],
    }

    device = torch.device(
        cfg["models"]["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not output_dir:
        output_dir = data_path.parent / f"{data_path.stem}_advanced_annotations"
    output_dir = Path(output_dir)
    labels_output_dir = output_dir / "labels_txt"
    json_output_dir = output_dir / "labels_json"
    vis_output_dir = output_dir / "visualizations"
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # intr = cfg["calib"]["intrinsics"]
    # camera_matrix = np.array(
    #     [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]
    # )
    # dist_coeffs = np.array(intr["distortion"])

    # old style
    # extr = cfg["calib"]["extrinsics"]
    # lidar_to_camera_tf = get_transform_matrix(extr["translation"], extr["rotation"])

    # ros2_numpy version
    extr = cfg["calib"]["extrinsics"]
    t = extr["translation"]
    q = extr["rotation"]
    translation_m = ros2_numpy.geometry.transformations.translation_matrix(t)
    rotation_m = ros2_numpy.geometry.transformations.quaternion_matrix(q)
    T_lidar_camera = np.dot(translation_m, rotation_m)
    lidar_to_cam_matrix = np.linalg.inv(T_lidar_camera)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=cfg["paths"]["yolo_model"],
        confidence_threshold=cfg["detection"]["confidence_threshold"],
        device=device,
    )

    # using CPU version as of now
    sam = sam_model_registry[cfg["models"]["sam_model_type"]]()
    state_dict = torch.load(cfg["paths"]["sam_checkpoint"], map_location="cpu")
    sam.load_state_dict(state_dict)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, text_color=sv.Color.BLACK)

    tracker_history = {}
    image_paths = sorted(
        [
            p
            for p in data_path.glob("*")
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]
    )

    for i, image_path in enumerate(image_paths):
        vis_fn = vis_output_dir / image_path.name
        json_fn = json_output_dir / f"{image_path.stem}.json"
        label_fn = labels_output_dir / f"{image_path.stem}.txt"
        if vis_fn.exists() and json_fn.exists() and label_fn.exists():
            print(f"skipping already processed frame : {image_path.name}")
            continue

        print(f"Processing frame {i+1}/{len(image_paths)}: {image_path.name}")

        try:
            frame = cv2.imread(str(image_path))
            pcd_file = pcd_path / f"{image_path.stem}.pcd"
            if not pcd_file.exists():
                print(
                    f"Warning: Point cloud not found for {image_path.name}. Skipping."
                )
                continue
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            points_3d_lidar = np.asarray(pcd.points)
            if frame is None:
                continue

            sahi_result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=cfg["sahi"]["slice_height"],
                slice_width=cfg["sahi"]["slice_width"],
                overlap_height_ratio=cfg["sahi"]["overlap_ratio"],
                overlap_width_ratio=cfg["sahi"]["overlap_ratio"],
            )
            xyxy_list = []
            confidence_list = []
            class_id_list = []

            if sahi_result.object_prediction_list:
                for pred in sahi_result.object_prediction_list:
                    xyxy_list.append(pred.bbox.to_xyxy())
                    confidence_list.append(pred.score.value)
                    class_id_list.append(pred.category.id)

            detections = sv.Detections(
                xyxy=np.array(xyxy_list),
                confidence=np.array(confidence_list),
                class_id=np.array(class_id_list).astype(int),
            )

            detections = detections[detections.class_id == 0]
            detections = detections.with_nms(
                threshold=cfg["detection"]["nms_threshold"]
            )

            tracked_detections = tracker.update_with_detections(detections)
            tracked_detections = tracked_detections[
                tracked_detections.tracker_id != None
            ]

            if len(tracked_detections) == 0:
                cv2.imwrite(str(vis_output_dir / image_path.name), frame)
                continue

            points_homogeneous = np.hstack(
                [points_3d_lidar, np.ones((points_3d_lidar.shape[0], 1))]
            )
            points_3d_cam = (lidar_to_cam_matrix @ points_homogeneous.T).T[:, :3]

            # normal image based projection
            # points_2d, _ = cv2.projectPoints(
            #     points_3d_cam, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
            # )
            # points_2d = points_2d.squeeze(axis=1)

            # using spherical(equirectangualar) projection
            x = points_3d_cam[:, 0]
            y = points_3d_cam[:, 1]
            z = points_3d_cam[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            points_2d = np.full(
                (x.shape[0], 2), -1.0
            )  # Initialize points_2d with an off-image value (-1, -1)
            valid_indices = r > 0.001

            # Calculate spherical coordinates only for valid points
            x_valid = x[valid_indices]
            y_valid = y[valid_indices]
            z_valid = z[valid_indices]
            r_valid = r[valid_indices]

            p_x = x_valid / r_valid
            p_y = y_valid / r_valid
            p_z = z_valid / r_valid

            # Clamp p_y to [-1, 1] to avoid domain errors in arcsin
            p_y = np.clip(p_y, -1.0, 1.0)
            phi = np.arcsin(p_y)
            theta = np.arctan2(p_x, p_z)

            H, W, _ = frame.shape
            # Calculate pixel coordinates for valid points
            u_coords = (theta * W / (2 * np.pi)) + (W / 2)
            v_coords = (phi * H / np.pi) + (H / 2)

            # Place the valid pixel coordinates into the main array
            points_2d[valid_indices] = np.vstack((u_coords, v_coords)).T

            sam_predictor.set_image(frame)
            masks_tensor, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=torch.tensor(tracked_detections.xyxy).to(device),
                multimask_output=False,
            )
            tracked_detections.mask = masks_tensor.cpu().numpy()  # .squeeze(axis=1)

            json_frame_data = []
            custom_labels_list = []

            # experimenting a mask reshaping functionality
            if len(tracked_detections) > 0:
                H, W, _ = frame.shape
                scale = 1024 / max(H, W)
                frame_for_sam = cv2.resize(frame, (int(W * scale), int(H * scale)))
                sam_predictor.set_image(frame_for_sam)

                scaled_boxes = tracked_detections.xyxy * scale
                masks_tensor, _, _ = sam_predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=torch.tensor(scaled_boxes).to(device),
                    multimask_output=False,
                )
                masks_np = masks_tensor.cpu().numpy()  # .squeeze(axis=1)
                num_detections = len(tracked_detections)
                final_masks = np.zeros((num_detections, H, W), dtype=bool)

                for idx, mask in enumerate(masks_np):
                    mask_2d = mask.squeeze()
                    resize_mask = cv2.resize(
                        mask_2d.astype(np.uint8),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    final_masks[idx] = resize_mask

                tracked_detections.mask = final_masks

            with open(
                labels_output_dir / f"{image_path.stem}.txt", "w", encoding="utf-8"
            ) as f_text:
                for idx in range(len(tracked_detections)):
                    bbox_2d = tracked_detections.xyxy[idx]
                    track_id = tracked_detections.tracker_id[idx]
                    class_id = tracked_detections.class_id[idx]
                    mask_2d_unresized = tracked_detections.mask[idx]

                    u = points_2d[:, 0]
                    v = points_2d[:, 1]
                    mask_in_box = (
                        (u >= bbox_2d[0])
                        & (u < bbox_2d[2])
                        & (v >= bbox_2d[1])
                        & (v < bbox_2d[3])
                    )
                    point_cluster = points_3d_lidar[mask_in_box]

                    center_2d = get_center_point(bbox_2d)
                    obj_status = "people.moving"
                    center_3d = np.array([0, 0, 0])
                    scale_3d = np.array([0, 0, 0])
                    angle_z = 0

                    json_obj = {}
                    if point_cluster.shape[0] >= 4:
                        cluster_pcd = o3d.geometry.PointCloud()
                        cluster_pcd.points = o3d.utility.Vector3dVector(point_cluster)

                        try:
                            plane_model, inliers = cluster_pcd.segment_plane(
                                distance_threshold=0.05, ransac_n=3, num_iterations=100
                            )
                            # The 'outliers' are the points NOT on the ground (i.e., the person)
                            outlier_cloud = cluster_pcd.select_by_index(
                                inliers, invert=True
                            )
                        except Exception as e:
                            print(f"  RANSAC failed for track {track_id}: {e}")
                            outlier_cloud = (
                                cluster_pcd  # Fallback to using the whole cluster
                            )

                        # Check if the outlier cloud (the person) still has enough points
                        if len(outlier_cloud.points) >= 4:
                            oriented_bbox_3d = outlier_cloud.get_oriented_bounding_box()
                            center_3d = oriented_bbox_3d.center
                            scale_3d = oriented_bbox_3d.extent
                            rotation_matrix_3d = oriented_bbox_3d.R
                            angle_z = np.arctan2(
                                rotation_matrix_3d[1, 0], rotation_matrix_3d[0, 0]
                            )

                        # Squeeze the mask to remove the singleton dimension (from 1xHxW to HxW)
                        # squeezed_mask = mask.squeeze()

                        # pixel based tracking
                        if track_id in tracker_history:
                            last_center_3d, last_center_2d, static_frames = (
                                tracker_history[track_id]
                            )
                            distance = math.dist(center_2d, last_center_2d)
                            distance_3d = math.dist(center_3d, last_center_3d)
                            if distance < cfg["tracking"]["movement_threshold_pixels"]:
                                static_frames += 1
                            else:
                                static_frames = 0

                            if (
                                static_frames
                                >= cfg["tracking"]["static_frame_count_threshold"]
                            ):
                                obj_status = "people.static"
                            tracker_history[track_id] = (
                                center_3d,
                                center_2d,
                                static_frames,
                            )
                        else:
                            tracker_history[track_id] = (center_3d, center_2d, 0)

                        # # trying velocity based tracking
                        # now = time.time()
                        # velocity_threshold = cfg['tracking'].get('velocity_threshold', 0.03) # meters per second
                        # obj_status = "people.moving"

                        # if track_id in tracker_history:
                        #     last_center_3d, last_time = tracker_history[track_id]
                        #     dt = now - last_time
                        #     if dt > 0.15:
                        #         disp = np.linalg.norm(np.array(center_3d) - np.array(last_center_3d))
                        #         velocity = disp / dt
                        #         if velocity < velocity_threshold:
                        #             obj_status = "people.static"
                        #     tracker_history[track_id] = (center_3d, now)
                        # else:
                        #     tracker_history[track_id] = (center_3d, now)

                    else:
                        # default for missing cluster
                        center_3d = np.array([0, 0, 0])
                        scale_3d = np.array([0, 0, 0])
                        angle_z = 0
                        obj_status = "people.static"  # default

                    json_obj = {
                        "obj_id": str(track_id),
                        "obj_type": obj_status,
                        "psr": {
                            "position": {
                                "x": float(center_3d[0]),
                                "y": float(center_3d[1]),
                                "z": float(center_3d[2]),
                            },
                            "rotation": {"x": 0, "y": 0, "z": float(angle_z)},
                            "scale": {
                                "x": float(scale_3d[0]),
                                "y": float(scale_3d[1]),
                                "z": float(scale_3d[2]),
                            },
                        },
                    }
                    json_frame_data.append(json_obj)
                    custom_labels_list.append(f"#{track_id} {obj_status.split('.')[1]}")

                    if np.any(mask_2d_unresized):
                        H, W, _ = frame.shape
                        mask_2d_resized = cv2.resize(
                            mask_2d_unresized.astype(np.uint8),
                            (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        polygons = sv.mask_to_polygons(mask_2d_resized)
                        if polygons:
                            segment = polygons[0] / np.array(
                                [frame.shape[1], frame.shape[0]]
                            )
                            segment_str = " ".join(map(str, segment.flatten()))
                            f_text.write(f"{class_id} {segment_str}\n")

            if json_frame_data:
                with open(
                    json_output_dir / f"{image_path.stem}.json", "w", encoding="utf-8"
                ) as f_json:
                    json.dump(json_frame_data, f_json, indent=2)

            if len(tracked_detections) > 0 and len(custom_labels_list) > 0:
                annotated_frame = frame.copy()
                annotated_frame = mask_annotator.annotate(
                    scene=annotated_frame, detections=tracked_detections
                )
                annotated_frame = box_annotator.annotate(
                    scene=annotated_frame, detections=tracked_detections
                )
                annotated_frame = label_annotator.annotate(
                    annotated_frame,
                    detections=tracked_detections,
                    labels=custom_labels_list,
                )
                cv2.imwrite(str(vis_output_dir / image_path.name), annotated_frame)

                print(f"Processing complete. Results saved in: {output_dir}")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory on frame {image_path.name}, skipping...")
                torch.cuda.empty_cache()
                time.sleep(10)  # small pause to let VRAM clear
                continue
            else:
                print(f"RuntimeError on {image_path.name}: {e}")
                continue

        except Exception as e:
            print(f"Exception processing {image_path.name}: {e}")
            continue


if __name__ == "__main__":
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
        default="config.yml",
        help="Path to the configuration YAML file. (Optional)",
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()

    track_and_annotate(
        data=args.data, pcd_dir=args.pcd_dir, config=config, output_dir=args.output_dir
    )
