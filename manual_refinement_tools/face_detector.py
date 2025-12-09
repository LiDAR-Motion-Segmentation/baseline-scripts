from tabnanny import verbose
import torch
from ultralytics import YOLO
from typing import List, Tuple
import logging

log = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, model_path: str, device: str, conf_thresh: float) -> None:
        self.device = self._validate_device(device)
        self.conf_thres = conf_thresh

        log.info(f"Loading YOLO model from {model_path} to {self.device}...")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise

    def _validate_device(self, request_device: str) -> str:
        if "cuda" in request_device and not torch.cuda.is_available():
            log.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return request_device

    def detect(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of bounding boxes (x, y, w, h).
        """
        results = self.model.predict(
            source=image_path, conf=self.conf_thres, device=self.device, verbose=False
        )

        boxes_xywh = []
        for result in results:
            bbox_list = result.boxes.xywh.cpu().numpy()
            for box in bbox_list:
                cx, cy, w, h = box
                # Convert Center-XY to Top-Left-XY
                x = int(cx - (w / 2))
                y = int(cy - (h / 2))
                boxes_xywh.append((x, y, int(w), int(h)))

        return boxes_xywh
