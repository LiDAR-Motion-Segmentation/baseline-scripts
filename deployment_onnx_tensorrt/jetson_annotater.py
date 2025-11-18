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
        self.camera_matrix, self.distortion_coeffs = self._load_intrinsics(self.config)

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
    def _post_process_yolo_output(
        output: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        padded_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ) -> np.ndarray:
        predictions = output[0].T  # Transpose (1, 84, N) to (N, 84)

        scores = np.max(predictions[:, 4:], axis=1)
        valid_preds = predictions[scores > conf_threshold]
        if len(valid_preds) == 0:
            return np.empty((0, 6))

        class_ids = np.argmax(valid_preds[:, 4:], axis=1)

        # Convert bounding box format (center_x, center_y, width, height) to (x1, y1, x2, y2)
        boxes = valid_preds[:, :4]
        boxes[:, 0] -= boxes[:, 2] / 2  # x1 = center_x - width / 2
        boxes[:, 1] -= boxes[:, 3] / 2  # y1 = center_y - height / 2
        boxes[:, 2] += boxes[:, 0]  # x2 = x1 + width
        boxes[:, 3] += boxes[:, 1]  # y2 = y1 + height

        # Scale boxes back to original image size
        boxes /= scale

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores[scores > conf_threshold].tolist(),
            conf_threshold,
            iou_threshold,
        )
        if len(indices) == 0:
            return np.empty((0, 6))

        final_boxes = boxes[indices.flatten()]
        final_scores = scores[scores > conf_threshold][indices.flatten()]
        final_class_ids = class_ids[indices.flatten()]
        results = np.hstack(
            (final_boxes, final_scores[:, np.newaxis], final_class_ids[:, np.newaxis])
        )

        return results[
            results[:, 5] == 0
        ]  # Filter for class_id 0 (person) if applicable
