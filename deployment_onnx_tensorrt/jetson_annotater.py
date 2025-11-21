import argparse
import cv2
import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R  # For quaternion to matrix
from typing import Dict, Any, Tuple, List, Optional
import onnxruntime as ort  # For ONNX/TensorRT inference
from tqdm import tqdm  # For progress bar


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    # scipy's Rotation expects (x, y, z, w)
    rotation = R.from_quat(q)
    return rotation.as_matrix()


class JetsonAnnotator:
    def __init__(self, config_path: str, debug_mode: bool = False):
        self.config = self._load_config(config_path)
        self.debug_mode = debug_mode
        self.image_h = None
        self.image_w = None
        print("Initializing JetsonAnnotator...")
        self.T_LIDAR_Camera = self._load_extrinsics(self.config)
        self.camera_matrix, self.dist_coeffs = self._load_intrinsics(self.config)

        self.yolo_session = self._create_onnx_session(
            self.config["models"]["yolo_onnx_path"]
        )
        self.sam_encoder_session = self._create_onnx_session(
            self.config["models"]["sam_encoder_onnx_path"]
        )
        self.sam_decoder_session = self._create_onnx_session(
            self.config["models"]["sam_decoder_onnx_path"]
        )

        print("JetsonAnnotator initialized successfully.")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config is None:
                    raise ValueError(
                        f"Config file '{config_path}' is empty or malformed."
                    )
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at '{config_path}'")
        except Exception as e:
            raise RuntimeError(f"Error loading config file: {e}")

    def _load_extrinsics(self, config: str) -> np.ndarray:
        try:
            T_lidar_cam_arr = np.array(config["T_lidar_camera"], dtype=np.float64)
            t = T_lidar_cam_arr[0:3]
            q = T_lidar_cam_arr[3:7]
            rotation_m_3x3 = quaternion_to_rotation_matrix(q)
            T_LIDAR_Camera = np.eye(4, dtype=np.float64)
            T_LIDAR_Camera[:3, :3] = rotation_m_3x3
            T_LIDAR_Camera[:3, 3] = t
            return T_LIDAR_Camera
        except KeyError:
            raise ValueError(
                "Error: 'T_lidar_camera' not found or malformed in config.yml. Expected [tx, ty, tz, qx, qy, qz, qw]."
            )
        except Exception as e:
            raise RuntimeError(f"Error parsing T_lidar_camera: {e}")

    def _load_intrinsics(self, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Loads camera intrinsic matrix and distortion coefficients."""
        try:
            intr = config["intrinsics"]
            camera_matrix = np.array(
                [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]],
                dtype=np.float64,
            )
            dist_coeffs = np.array(intr["distortion"], dtype=np.float64)

            # Store image dimensions from intrinsics for SAM resizing
            self.image_h = intr.get("height", None)
            self.image_w = intr.get("width", None)
            if self.image_h is None or self.image_w is None:
                print(
                    "Warning: Image height/width not found in intrinsics. Will use actual image dimensions for SAM, which might be less optimal."
                )
            return camera_matrix, dist_coeffs
        except KeyError:
            raise ValueError(
                "Error: 'intrinsics' section (with fx, fy, cx, cy, distortion, height, width) not found or malformed in config.yml."
            )
        except Exception as e:
            raise RuntimeError(f"Error parsing intrinsics: {e}")

    def _create_onnx_session(self, model_path: str) -> ort.InferenceSession:
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"ONNX/TensorRT model not found: {model_path}")

        print(f"Loading ONNX/TensorRT model: {model_path}")
        try:
            providers = [
                self.config["models"].get("onnx_provider", "CPUExecutionProvider")
            ]
            # For TensorRT specific engines, you might need TensorrtExecutionProvider
            # onnxruntime-gpu usually enables CUDAExecutionProvider by default if available.
            session = ort.InferenceSession(str(model_path), providers=providers)
            print(f"  Model loaded with providers: {session.get_providers()}")
            return session
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model {model_path}: {e}")

    @staticmethod
    def _preprocess_yolo_image(
        image: np.ndarray, target_size: int = 640
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w, _ = image.shape
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded_img[:new_h, :new_w, :] = resized_img

        input_img = padded_img.transpose(2, 0, 1)
        input_img = np.ascontiguousarray(input_img, dtype=np.float32) / 255.0
        input_img = np.expand_dims(input_img, 0)  # Add batch dimension (1, C, H, W)
        return input_img, scale, (new_w, new_h)

    @staticmethod
    def _postprocess_yolo_output(
        output: np.ndarray,
        original_shape: Tuple[int, int],  # (orig_h, orig_w)
        scale: float,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,  # Adjusted default IOU for better filtering
        class_id_filter: Optional[
            int
        ] = 0,  # Filter for 'person' class (YOLOv8 default class 0)
    ) -> np.ndarray:
        """
        Postprocesses YOLOv8 ONNX output to get bounding boxes.
        Applies NMS and scales boxes back to original image size.
        Output format: (num_boxes, [x1, y1, x2, y2, confidence, class_id])
        """
        predictions = output[0].T  # Transpose (1, 84, N) to (N, 84)

        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)  # Max score across all classes
        valid_preds_indices = np.where(scores > conf_threshold)[0]
        if len(valid_preds_indices) == 0:
            return np.empty((0, 6), dtype=np.float32)

        valid_preds = predictions[valid_preds_indices]
        scores = scores[valid_preds_indices]

        # Get class ID
        class_ids = np.argmax(valid_preds[:, 4:], axis=1)

        # Convert bounding box format (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = valid_preds[:, :4]
        boxes[:, 0] -= boxes[:, 2] / 2  # x1 = center_x - width / 2
        boxes[:, 1] -= boxes[:, 3] / 2  # y1 = center_y - height / 2
        boxes[:, 2] += boxes[:, 0]  # x2 = x1 + width
        boxes[:, 3] += boxes[:, 1]  # y2 = y1 + height

        # Scale boxes back to original image size
        boxes /= scale

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
        )
        if len(indices) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # might throw an error here
        final_indices = indices.flatten()
        final_boxes = boxes[final_indices]
        final_scores = scores[final_indices]
        final_class_ids = class_ids[final_indices]

        results = np.hstack(
            (final_boxes, final_scores[:, np.newaxis], final_class_ids[:, np.newaxis])
        )

        # Filter by a specific class ID if requested (e.g., 'person' = 0)
        if class_id_filter is not None:
            results = results[results[:, 5] == class_id_filter]

        return results.astype(np.float32)

    def _preprocess_sam_image(
        self, image: np.ndarray, target_size: int = 1024
    ) -> Tuple[np.ndarray, float]:
        h, w, _ = image.shape
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full(
            (target_size, target_size, 3), 0, dtype=np.uint8
        )  # sam uses 0 padding
        padded_img[:new_h, :new_w, :] = resized_img

        # HWC to CHW, SAM-specific normalization
        input_img = padded_img.transpose(2, 0, 1)
        input_img = np.ascontiguousarray(input_img, dtype=np.float32)
        input_img = (
            input_img - np.array([123.675, 116.28, 103.53]).reshape(3, 1, 1)
        ) / np.array([58.395, 57.12, 57.375]).reshape(
            3, 1, 1
        )  # SAM normalization
        input_img = np.expand_dims(input_img, 0)  # Add batch dimension
        return input_img, scale

    @staticmethod
    def _postprocess_sam_mask(
        mask_output: np.ndarray,
        original_shape: Tuple[int, int],  # (H_orig, W_orig)
        input_scale: float,
        target_size: int = 1024,
    ) -> np.ndarray:
        mask = cv2.resize(
            mask_output[0, 0, :, :],  # Remove batch and channel dims, get 256x256
            (target_size, target_size),  # Upscale to SAM's internal processing size
            interpolation=cv2.INTER_LINEAR,
        )
        intermediate_h = int(original_shape[0] * input_scale)
        intermediate_w = int(original_shape[1] * input_scale)
        mask = mask[:intermediate_h, :intermediate_w]

        final_mask = cv2.resize(
            mask,
            (original_shape[1], original_shape[0]),  # (width, height)
            interpolation=cv2.INTER_LINEAR,
        )

        return final_mask > 0.0  # convert to boolean mask

    def _get_yolo_detections(self, image: np.ndarray) -> np.ndarray:
        input_img, scale, original_hw = self._preprocess_yolo_image(image)
        ort_inputs = {self.yolo_session.get_inputs()[0].name: input_img}
        ort_outputs = self.yolo_session.run(None, ort_inputs)
        detections = self._postprocess_yolo_output(ort_outputs[0], original_hw, scale)
        return detections  # (N, [x1, y1, x2, y2, confidence, class_id])

    def _get_sam_masks(self, image: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        if len(bboxes) == 0:
            return []

        target_h, target_w = self.image_h, self.image_w
        if target_h is None or target_w is None:
            target_h, target_w = (
                image.shape[0],
                image.shape[1],
            )  # Fallback to actual image dimensions

        input_img, input_scale = self._preprocess_sam_image(
            image, target_size=(target_h, target_w)
        )  # SAM expects square input

        encoder_input_name = self.sam_encoder_session.get_inputs()[0].name
        image_embedding = self.sam_encoder_session.run(
            None, {encoder_input_name: input_img}
        )[
            0
        ]  # (1, 256, 64, 64) for large SAM

        all_masks = []
        for bbox in bboxes:
            scaled_bbox = bbox * input_scale

            ort_inputs_decoder = {
                "image_embeddings": image_embedding,
                "point_coords": np.array(
                    [[[0.0, 0.0]]], dtype=np.float32
                ),  # No point prompt, so just a dummy point
                "point_labels": np.array(
                    [[-1.0]], dtype=np.float32
                ),  # -1 indicates no point label
                "has_mask_input": np.array([0.0], dtype=np.float32),  # 0.0 for False
                "orig_im_size": np.array(
                    [image.shape[0], image.shape[1]], dtype=np.float32
                ),  # Original image size
                "mask_input": np.zeros(
                    (1, 1, 256, 256), dtype=np.float32
                ),  # Initial empty mask input
                "box_coords": np.expand_dims(scaled_bbox, axis=0).astype(
                    np.float32
                ),  # Single box (1, 4)
            }

            ##################################################################################################################################

            # Run SAM Decoder
            # The outputs typically are: masks, iou_predictions, low_res_masks
            masks_output, iou_pred, low_res_masks = self.sam_decoder_session.run(
                None, ort_inputs_decoder
            )

            # Postprocess mask to original image size. SAM's low_res_masks are usually 256x256
            mask_decoded = self._postprocess_sam_mask(
                low_res_masks, image.shape[:2], input_scale
            )
            all_masks.append(mask_decoded)

        return all_masks

    def _project_lidar_to_image(
        self, xyz_points: np.ndarray, image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Projects 3D LiDAR points to 2D image coordinates using the plumb bob (pinhole) model.
        Returns:
            points_2d (np.ndarray): (N_valid, 2) array of projected pixel coordinates.
            indices_forward (np.ndarray): (N_valid,) array of original indices of points.
            on_image_mask (np.ndarray): (N_forward,) boolean mask of points on image.
        """
        h, w = image_shape[:2]

        # 1. Transform points from LiDAR space to Camera 3D space
        points_homogeneous = np.hstack([xyz_points, np.ones((xyz_points.shape[0], 1))])
        points_camera_frame = (self.T_LIDAR_Camera @ points_homogeneous.T).T[:, :3]

        # 2. Filter out points *behind* the camera (z <= 0)
        z_forward_mask = points_camera_frame[:, 2] > 0
        points_3d_forward = points_camera_frame[z_forward_mask]

        original_indices = np.arange(len(xyz_points))
        indices_forward = original_indices[z_forward_mask]

        points_2d = None
        if len(points_3d_forward) > 0:
            points_2d, _ = cv2.projectPoints(
                points_3d_forward,
                np.zeros(3),
                np.zeros(3),  # rvec, tvec (already in camera frame)
                self.camera_matrix,
                self.dist_coeffs,
            )
            points_2d = points_2d.squeeze(axis=1)  # (N, 1, 2) -> (N, 2)

        if points_2d is None or len(points_2d) == 0:
            return (
                np.empty((0, 2), dtype=np.float32),
                np.empty(0, dtype=int),
                np.empty(0, dtype=bool),
            )

        # 3. Filter points that are within the image boundaries
        on_image_mask = (
            (points_2d[:, 0] >= 0)
            & (points_2d[:, 0] < w)
            & (points_2d[:, 1] >= 0)
            & (points_2d[:, 1] < h)
        )

        return points_2d, indices_forward, on_image_mask

    def process_frame(
        self,
        pcd_path: Path,
        image_path: Path,
        output_dir: Path,
    ):
        """
        Processes a single frame:
        1. Loads LiDAR data and image.
        2. Runs YOLOv8 ONNX for 2D object detection (person).
        3. Runs SAM ONNX for mask generation based on YOLO boxes.
        4. Projects 3D LiDAR points to 2D image.
        5. Filters 3D points by 2D masks to get segmented 3D objects.
        6. Saves segmented 3D points (Numpy) and colored PCDs.
        """
        frame_stem = pcd_path.stem
        # print(f"Processing frame: {frame_stem}") # Keep this less verbose for tqdm

        # --- 1. Load Data ---
        pc_data = o3d.io.read_point_cloud(str(pcd_path))
        xyz_points = np.asarray(pc_data.points, dtype=np.float64)  # Ensure float64
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Warning: Could not load image at {image_path}. Skipping.")
            return

        h, w, _ = image.shape

        yolo_results = self._get_yolo_detections(image)

        if len(yolo_results) == 0:
            # print(f"No persons detected in image for {frame_stem}. Skipping 3D segmentation.")
            # Still save an annotated image showing no detections if desired
            cv2.imwrite(str(output_dir / "sam_image" / f"{frame_stem}.png"), image)
            # Create empty files to indicate no detections if that's your protocol
            np.save(
                str(output_dir / "sam_lidar_segmented" / f"{frame_stem}.npy"),
                np.empty((0, 3), dtype=np.float64),
            )
            o3d.io.write_point_cloud(
                str(output_dir / "sam_lidar" / f"{frame_stem}.pcd"),
                o3d.geometry.PointCloud(),
            )
            return

        bboxes_xyxy = yolo_results[:, :4]  # Extract only bounding boxes

        all_masks = self._get_sam_masks(image, bboxes_xyxy)

        if not all_masks:
            # print(f"SAM did not generate masks for detections in {frame_stem}. Skipping 3D segmentation.")
            cv2.imwrite(str(output_dir / "sam_image" / f"{frame_stem}.png"), image)
            np.save(
                str(output_dir / "sam_lidar_segmented" / f"{frame_stem}.npy"),
                np.empty((0, 3), dtype=np.float64),
            )
            o3d.io.write_point_cloud(
                str(output_dir / "sam_lidar" / f"{frame_stem}.pcd"),
                o3d.geometry.PointCloud(),
            )
            return

        # Combine all masks into a single segmentation_mask
        segmentation_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_idx, mask_bool in enumerate(all_masks):
            # Assign a unique ID or just set to 1 for all person masks
            # Here we just set to 1, indicating 'is a person' area.
            segmentation_mask[mask_bool] = 1

        # Create and save 2D visualization image (YOLO boxes + SAM masks)
        overlay = image.copy()
        for i, bbox in enumerate(bboxes_xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            if all_masks[i] is not None:
                # Apply mask overlay
                color_mask = np.zeros_like(image, dtype=np.uint8)
                color_mask[all_masks[i]] = [0, 0, 255]  # Red mask
                overlay = cv2.addWeighted(overlay, 1, color_mask, 0.4, 0)

        final_image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)  # Blend
        cv2.imwrite(str(output_dir / "sam_image" / f"{frame_stem}.png"), final_image)

        # Project 3D LiDAR Points to 2D Image (Pinhole Model)
        points_2d, indices_forward, on_image_mask = self._project_lidar_to_image(
            xyz_points, image.shape[:2]
        )

        if len(points_2d) == 0:
            # print(f"No LiDAR points projected onto image for {frame_stem}. Skipping 3D segmentation.")
            np.save(
                str(output_dir / "sam_lidar_segmented" / f"{frame_stem}.npy"),
                np.empty((0, 3), dtype=np.float64),
            )
            o3d.io.write_point_cloud(
                str(output_dir / "sam_lidar" / f"{frame_stem}.pcd"),
                o3d.geometry.PointCloud(),
            )
            return

        if self.debug_mode:
            debug_image = final_image.copy()  # Use the annotated 2D image
            valid_pixels_on_image = points_2d[on_image_mask].astype(int)
            for px_x, px_y in valid_pixels_on_image:
                cv2.circle(debug_image, (px_x, px_y), 1, (0, 255, 0), -1)  # Green dots
            cv2.imwrite(
                str(output_dir / "debug_projection" / f"{frame_stem}_debug.png"),
                debug_image,
            )

        # Get the 2D pixel coordinates of points that are on the image plane
        valid_pixels = points_2d[on_image_mask].astype(int)

        # Get the *original* indices corresponding to these valid pixels
        indices_on_image = indices_forward[on_image_mask]

        person_indices = np.array([], dtype=int)
        if len(valid_pixels) > 0:
            # Look up the mask value for each valid pixel
            mask_values = segmentation_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
            person_mask_on_image = (
                mask_values == 1
            )  # True where the mask indicates a person

            person_indices = indices_on_image[person_mask_on_image]
        # else:
        # print(f"Warning: No LiDAR points projected onto the image boundaries for {frame_stem}.") # Less critical now

        person_points = xyz_points[person_indices]
        np.save(
            str(output_dir / "sam_lidar_segmented" / f"{frame_stem}.npy"), person_points
        )

        colors = np.full(
            (len(xyz_points), 3), [0.0, 1.0, 0.0]
        )  # Default: green for non-person
        colors[person_indices] = [1.0, 0.0, 0.0]  # Red for person points

        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(xyz_points)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(
            str(output_dir / "sam_lidar" / f"{frame_stem}.pcd"), colored_pcd
        )
        # print(f"Finished processing and saving results for {frame_stem}.") # Less verbose for tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Jetson-optimized 3D annotation pipeline with ONNX/TensorRT models."
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
        help="Frame offset between PCD and image files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for projection visualization and more verbose output.",
    )
    args = parser.parse_args()

    # --- Setup Output Directories ---
    output_dir = Path(args.output_dir)
    (output_dir / "sam_lidar").mkdir(parents=True, exist_ok=True)
    (output_dir / "sam_image").mkdir(parents=True, exist_ok=True)
