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
    sam_model_type: str = "vit_l"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


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

        # If image is loaded with cv2: convert from BGR to RGB
        # if image.shape[2] == 3 and image.dtype == np.uint8:
        #     # image must be HWC np.uint8 RGB for most DINO forks
        #     pass
        # else:
        #     raise ValueError("Image must be RGB uint8 (H,W,3) for GroundingDINO.")

        if isinstance(image, np.ndarray):
            # If RGB image shape (H, W, 3), convert to CHW and float32
            image_tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.config.device)
        else:
            image_tensor = image.to(self.config.device)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
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

    def _setup_output_dirs(self) -> None:
        for cam in self.cameras:
            (self.output_dir / cam.name / "yolo_boxes").mkdir(
                parents=True, exist_ok=True
            )
            (self.output_dir / cam.name / "sam_masks").mkdir(
                parents=True, exist_ok=True
            )
            (self.output_dir / cam.name / "visualizations").mkdir(
                parents=True, exist_ok=True
            )

    def process_frame(self, image_path: Path, camera_name: str) -> None:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load {image_path}")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_name = image_path.stem
        boxes, scores, labels = self.detector.detect(image_rgb, self.text_prompt)

        if len(boxes) == 0:
            print(f"No detections for {camera_name}/{frame_name}")
            return

        masks = self.segmentor.segment(image_rgb, boxes)

        yolo_path = self.output_dir / camera_name / "yolo_boxes" / f"{frame_name}.txt"
        YOLOWriter.write_yolo_annotations(boxes, image.shape[:2], yolo_path)

        mask_path = self.output_dir / camera_name / "sam_masks" / f"{frame_name}.txt"
        PolygonWriter.write_polygon_annotations(masks, image.shape[:2], mask_path)

        self._visualize_detections(image, boxes, masks, camera_name, frame_name)
        print(f"[{camera_name}] Processed {frame_name}: {len(boxes)} detections")

    def _visualize_detections(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray,
        camera_name: str,
        frame_name: str,
    ) -> None:

        vis = image.copy()
        for mask in masks:
            color = np.random.randint(0, 255, 3).tolist()
            vis[mask > 0] = vis[mask > 0] * 0.6 + np.array(color) * 0.4

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        vis_path = (
            self.output_dir / camera_name / "visualizations" / f"{frame_name}.jpg"
        )
        cv2.imwrite(str(vis_path), vis)

    def run(self) -> None:
        for cam in self.cameras:
            print(f"\nProcessing camera: {cam.name}")
            image_files = sorted(cam.image_dir.glob("*.png")) + sorted(
                cam.image_dir.glob("*.jpg")
            )
            for img_path in image_files:
                self.process_frame(img_path, cam.name)


def parse_camera_args(camera_args: List[str]) -> List[CameraConfig]:
    cameras = []
    for arg in camera_args:
        if ":" not in arg:
            raise ValueError(f"Camera arg must be 'name:path', got: {arg}")
        name, path = name, path = arg.split(":", 1)
        cameras.append(CameraConfig(name, Path(path)))
    return cameras


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Grounding DINO + SAM2 Pipeline"
    )
    parser.add_argument(
        "--camera_dirs",
        nargs="+",
        required=True,
        help="Camera directories as name:path (e.g., cam1:/data/cam1)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--grounding_dino_config",
        type=str,
        required=True,
        help="Path to Grounding DINO config file",
    )
    parser.add_argument(
        "--grounding_dino_checkpoint",
        type=str,
        required=True,
        help="Path to Grounding DINO checkpoint",
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="Path to SAM2 checkpoint"
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="person. people.",
        help="Text prompt for Grounding DINO",
    )

    args = parser.parse_args()
    cameras = parse_camera_args(args.camera_dirs)

    model_config = ModelConfig(
        grounding_dino_config=Path(args.grounding_dino_config),
        grounding_dino_checkpoint=Path(args.grounding_dino_checkpoint),
        sam_checkpoint=Path(args.sam_checkpoint),
    )

    pipeline = MultiCameraDetectionPipeline(
        cameras=cameras,
        model_config=model_config,
        output_dir=Path(args.output_dir),
        text_prompt=args.text_prompt,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
