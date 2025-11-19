"""
Usage:
    python grounding_dino_sam2_multicam.py \
        --camera_dirs camera1:/path/to/cam1 camera2:/path/to/cam2 ... \
        --output_dir ./output \
        --text_prompt "person. people."
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
import torch
from groundingdino.util.inference import load_model, predict
from segment_anything_hq import sam_model_registry, SamPredictor


@dataclass
class CameraConfig:
    name: str
    image_dir: Path

    def __post_init__(self):
        if not self.image_dir.exists():
            raise ValueError(f"Camera image directory does not exist: {self.image_dir}")


@dataclass
class ModelConfig:
    grounding_dino_config: Path
    grounding_dino_checkpoint: Path
    sam_checkpoint: Path
    sam_model_type: str = "vit_h"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GroundingDINODetector:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = load_model(
            str(config.grounding_dino_config),
            str(config.grounding_dino_checkpoint),
            device=config.device,
        )

    def detect(
        self, image: np.ndarray, text_prompt: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        # boxes: (N, 4) array in xyxy format
        # scores: (N,) confidence scores
        # labels: List of label strings
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            device=self.config.device,
        )
        # Convert from normalized [0,1] to pixel coords if needed
        h, w = image.shape[:2]
        boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
        return boxes.cpu().numpy(), logits.cpu().numpy(), phrases


class SAM2Segmentor:
    def __init__(self, config: ModelConfig):
        self.config = config
        sam = sam_model_registry[config.sam_model_type](
            checkpoint=str(config.sam_checkpoint)
        )
        sam.to(device=config.device)
        self.predictor = SamPredictor(sam)

    def segment(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Args:
            image: RGB image (H, W, 3)
            boxes: (N, 4) boxes in xyxy format

        Returns:
            masks: (N, H, W) boolean masks
        """
        self.predictor.set_image(image)

        # Convert boxes to SAM format (N, 4) tensor
        boxes_tensor = torch.tensor(boxes, device=self.config.device)

        # Batch predict
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=boxes_tensor,
            multimask_output=False,
        )

        # Return as numpy boolean array (N, H, W)
        return masks.squeeze(1).cpu().numpy()


class YOLOWriter:
    @staticmethod
    def write_yolo_annotations(
        boxes: np.ndarray,
        image_shape: Tuple[int, int],
        output_path: Path,
        class_id: int = 0,
    ) -> None:
        # Write YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized)
        h, w = image_shape
        with open(output_path, "w") as f:
            for box in boxes:
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                f.write(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )


class PolygonWriter:
    @staticmethod
    def mask_to_polygon(mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # get the largest contour
        largest = max(contours, key=cv2.contourArea)
        return largest.squeeze()

    @staticmethod
    def write_polygon_annotations(
        masks: np.ndarray,
        image_shape: Tuple[int, int],
        output_path: Path,
        class_id: int = 0,
    ) -> None:
        # Write polygon format: <class_id> <x1> <y1> <x2> <y2> ... (normalized)
        h, w = image_shape
        with open(output_path, "w") as f:
            for mask in masks:
                poly = PolygonWriter.mask_to_polygon(mask)
                if poly is None or len(poly) < 3:
                    continue
                # normalize coordinates
                poly_norm = poly / np.array([w, h])
                coords_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly_norm])
                f.write(f"{class_id} {coords_str}\n")


class MultiCameraDetectionPipeline:
    def __init__(
        self,
        cameras: List[CameraConfig],
        model_config: ModelConfig,
        output_dir: Path,
        text_prompt: str = "person. people.",
    ):
        self.cameras = cameras
        self.output_dir = output_dir
        self.text_prompt = text_prompt

        print("Loading Grounding DINO...")
        self.detector = GroundingDINODetector(model_config)
        print("Loading SAM2...")
        self.segmentor = SAM2Segmentor(model_config)

        self._setup_output_dirs()
