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
