import argparse
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import json
from collections import defaultdict
import gc
import time
import os
import numpy as np
import cv2
import torch

# Suppress verbose logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not available, will use YOLOv8 only")

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: YOLOv8 not available, will use MediaPipe only")

from tqdm import tqdm

import os


def setup_logging(output_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with file and console handlers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("FaceBlurring")
    logger.handlers.clear()  # Clear any existing handlers
    logger.setLevel(level)

    # File handler
    fh = logging.FileHandler(output_dir / "face_blurring.log")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s"
    )
    fh.setFormatter(fh_formatter)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter("%(levelname)s - %(message)s")
    ch.setFormatter(ch_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


@dataclass
class FaceDetection:
    """Represents a single detected face."""

    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float  # 0-1
    detector_source: str  # "mediapipe" or "yolov8"
    face_quality: float = 0.0
    blurred: bool = False


@dataclass
class FrameBlurringResult:
    """Result of processing one image."""

    image_path: Path
    num_faces_detected: int = 0
    num_faces_blurred: int = 0
    processing_time: float = 0.0
    success: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON."""
        return {
            "image_path": str(self.image_path),
            "num_faces_detected": self.num_faces_detected,
            "num_faces_blurred": self.num_faces_blurred,
            "processing_time": self.processing_time,
            "success": self.success,
            "error": self.error_message,
        }


class MediaPipeFaceDetector:
    """
    Thread-safe MediaPipe face detector.

    IMPORTANT: MediaPipe is thread-safe but we keep one instance per process.
    """

    def __init__(self, min_detection_confidence: float = 0.7):
        """Initialize MediaPipe (once per process)."""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not installed")

        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # Full-range model
            min_detection_confidence=min_detection_confidence,
        )
        self.logger = logging.getLogger("FaceBlurring")

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces in image."""
        try:
            h, w, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = self.detector.process(image_rgb)

            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    confidence = detection.score[0] if detection.score else 0.5

                    faces.append(
                        FaceDetection(
                            bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
                            confidence=confidence,
                            detector_source="mediapipe",
                        )
                    )

            return faces
        except Exception as e:
            self.logger.debug(f"MediaPipe detection error: {e}")
            return []


class YOLOv8FaceDetector:
    """
    YOLOv8 face detector with proper GPU memory management.

    Key difference: Force CPU inference to avoid CUDA context conflicts.
    """

    def __init__(
        self,
        model_path: str = "/scratch/soumo_roy/baseline-scripts/weights/yolov11l-face.pt",
        device: str = "cuda:0",
    ):
        """
        Initialize YOLOv8 face detector.

        Args:
            model_path: Path to model weights
            device: "cpu" (recommended) or "cuda"
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("YOLOv8 not installed")

        self.logger = logging.getLogger("FaceBlurring")
        self.device = device

        try:
            self.model = YOLO(model_path)
            # Force to specified device
            self.model.to(device)
            self.logger.info(f"YOLOv8 loaded on device: {device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLOv8: {e}")
            raise

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """Detect faces using YOLOv8."""
        try:
            with torch.no_grad():
                results = self.model.predict(
                    image,
                    conf=0.4,
                    device=self.device,
                    verbose=False,
                    half=False,  # Use FP32 for stability
                )

            faces = []
            for result in results:
                for detection in result.boxes:
                    x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(detection.conf)

                    faces.append(
                        FaceDetection(
                            bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
                            confidence=confidence,
                            detector_source="yolov8",
                        )
                    )

            return faces
        except Exception as e:
            self.logger.debug(f"YOLOv8 detection error: {e}")
            return []

    def cleanup(self):
        """Cleanup GPU memory after inference."""
        if self.device == "cuda:0":
            torch.cuda.empty_cache()
            gc.collect()


class FaceQualityAssessor:
    """Assess detected face quality."""

    MIN_FACE_SIZE = 20
    MAX_FACE_SIZE = 500
    BLUR_THRESHOLD = 100.0

    @staticmethod
    def assess_quality(image: np.ndarray, face: FaceDetection) -> float:
        """Compute quality score (0-1)."""
        x1, y1, x2, y2 = face.bbox
        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return 0.0

        h_face, w_face = face_crop.shape[:2]
        face_size = h_face * w_face

        # Size check
        if face_size < FaceQualityAssessor.MIN_FACE_SIZE**2:
            return 0.0
        if face_size > FaceQualityAssessor.MAX_FACE_SIZE**2:
            return 0.3

        # Aspect ratio
        aspect_ratio = max(w_face, h_face) / (min(w_face, h_face) + 1e-6)
        if aspect_ratio > 2.0:
            return 0.2

        # Blur detection
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = 1.0 if laplacian_var >= FaceQualityAssessor.BLUR_THRESHOLD else 0.5

        # Combined score
        size_score = 1.0
        ratio_score = 1.0 if aspect_ratio < 1.5 else 0.7
        conf_score = face.confidence

        quality = (
            0.4 * blur_score + 0.3 * size_score + 0.2 * ratio_score + 0.1 * conf_score
        )

        face.face_quality = quality
        return quality


class FaceBlurringEngine:
    """High-precision face blurring."""

    BLUR_STRENGTH_MAP = {
        "conservative": 21,
        "moderate": 51,
        "aggressive": 101,
    }

    @staticmethod
    def blur_face(
        image: np.ndarray,
        face: FaceDetection,
        strategy: str = "gaussian",
        blur_strength: str = "aggressive",
        expansion_ratio: float = 1.1,
    ) -> np.ndarray:
        """Blur a detected face region."""
        x1, y1, x2, y2 = face.bbox
        h, w = image.shape[:2]

        # Expand bounding box
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1

        new_width = int(width * expansion_ratio)
        new_height = int(height * expansion_ratio)

        x1_exp = max(0, int(center_x - new_width / 2))
        y1_exp = max(0, int(center_y - new_height / 2))
        x2_exp = min(w, int(center_x + new_width / 2))
        y2_exp = min(h, int(center_y + new_height / 2))

        image_copy = image.copy()
        face_region = image_copy[y1_exp:y2_exp, x1_exp:x2_exp]

        if face_region.size == 0:
            return image_copy

        # Apply blur
        if strategy == "gaussian":
            kernel_size = FaceBlurringEngine.BLUR_STRENGTH_MAP.get(blur_strength, 51)
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            blurred = cv2.GaussianBlur(face_region, (kernel_size, kernel_size), 0)

        elif strategy == "pixelate":
            pixel_size = {
                "conservative": 4,
                "moderate": 8,
                "aggressive": 16,
            }.get(blur_strength, 8)

            h_region, w_region = face_region.shape[:2]
            small = cv2.resize(
                face_region, (w_region // pixel_size, h_region // pixel_size)
            )
            blurred = cv2.resize(
                small, (w_region, h_region), interpolation=cv2.INTER_NEAREST
            )

        elif strategy == "mosaic":
            pixel_size = 12
            h_region, w_region = face_region.shape[:2]
            small = cv2.resize(
                face_region, (w_region // pixel_size, h_region // pixel_size)
            )
            blurred = cv2.resize(
                small, (w_region, h_region), interpolation=cv2.INTER_NEAREST
            )
            blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

        else:
            blurred = face_region

        image_copy[y1_exp:y2_exp, x1_exp:x2_exp] = blurred
        face.blurred = True

        return image_copy


class PrivacyPreservingFaceBlurrer:
    """
    Production-grade face blurring pipeline.

    KEY DESIGN: Sequential processing with GPU cleanup between images.
    This avoids segmentation faults from concurrent GPU access.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        blur_strategy: str = "gaussian",
        blur_strength: str = "aggressive",
        use_ensemble: bool = True,
        device: str = "cuda:0",
        use_mediapipe: bool = True,
        use_yolov8: bool = True,
    ):
        """Initialize the pipeline."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.blur_strategy = blur_strategy
        self.blur_strength = blur_strength
        self.device = device

        # Setup logging
        self.logger = setup_logging(self.output_dir)

        self.logger.info("=" * 70)
        self.logger.info("Privacy-Preserving Face Blurring Pipeline")
        self.logger.info("=" * 70)
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Strategy: {blur_strategy}, Strength: {blur_strength}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Use ensemble: {use_ensemble}")

        # Create output directories
        (self.output_dir / "blurred").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metadata").mkdir(parents=True, exist_ok=True)

        # Initialize detectors
        self.mediapipe_detector = None
        self.yolov8_detector = None
        self.use_ensemble = use_ensemble

        if use_mediapipe and MEDIAPIPE_AVAILABLE:
            try:
                self.mediapipe_detector = MediaPipeFaceDetector()
                self.logger.info("✓ MediaPipe face detector loaded")
            except Exception as e:
                self.logger.warning(f"✗ MediaPipe failed: {e}")

        if use_yolov8 and YOLO_AVAILABLE:
            try:
                # Force YOLOv8 to CPU to avoid GPU conflicts
                yolo_device = "cuda" if use_mediapipe else device
                self.yolov8_detector = YOLOv8FaceDetector(device=yolo_device)
                self.logger.info("✓ YOLOv8 face detector loaded")
            except Exception as e:
                self.logger.warning(f"✗ YOLOv8 failed: {e}")

        if not self.mediapipe_detector and not self.yolov8_detector:
            raise RuntimeError("No face detectors available!")

        self.quality_assessor = FaceQualityAssessor()
        self.blurring_engine = FaceBlurringEngine()

        # Statistics
        self.stats = {
            "total_images": 0,
            "successful": 0,
            "failed": 0,
            "total_faces_detected": 0,
            "total_faces_blurred": 0,
            "processing_time": 0.0,
        }

    def _detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces using available detectors.

        Strategy:
        1. Try MediaPipe (fast, accurate for frontal/profile)
        2. Try YOLOv8 (catches extreme angles, small faces)
        3. Deduplicate overlapping detections
        """
        all_faces = []

        # Primary detector
        if self.mediapipe_detector:
            faces_mp = self.mediapipe_detector.detect(image)
            all_faces.extend(faces_mp)

        # Fallback detector
        if self.use_ensemble and self.yolov8_detector:
            faces_yolo = self.yolov8_detector.detect(image)
            all_faces.extend(faces_yolo)
            self.yolov8_detector.cleanup()

        if not all_faces:
            return []

        # Deduplicate using IoU
        sorted_faces = sorted(all_faces, key=lambda f: f.confidence, reverse=True)
        deduplicated = []

        for face in sorted_faces:
            is_duplicate = False
            for existing in deduplicated:
                x1_a, y1_a, x2_a, y2_a = face.bbox
                x1_b, y1_b, x2_b, y2_b = existing.bbox

                inter_x1 = max(x1_a, x1_b)
                inter_y1 = max(y1_a, y1_b)
                inter_x2 = min(x2_a, x2_b)
                inter_y2 = min(y2_a, y2_b)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    area_a = (x2_a - x1_a) * (y2_a - y1_a)
                    area_b = (x2_b - x1_b) * (y2_b - y1_b)
                    union_area = area_a + area_b - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > 0.3:
                        is_duplicate = True
                        if face.confidence > existing.confidence:
                            deduplicated.remove(existing)
                            deduplicated.append(face)
                        break

            if not is_duplicate:
                deduplicated.append(face)

        return deduplicated

    def process_image(self, image_path: Path) -> FrameBlurringResult:
        """Process a single image sequentially."""
        start_time = time.time()
        result = FrameBlurringResult(image_path=image_path)

        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                result.error_message = "Failed to load image"
                return result

            # Detect faces
            faces = self._detect_faces(image)
            result.num_faces_detected = len(faces)

            if len(faces) == 0:
                result.success = True
                # Save unmodified image
                output_path = self.output_dir / "blurred" / image_path.name
                cv2.imwrite(str(output_path), image)
                result.processing_time = time.time() - start_time
                return result

            # Filter by quality
            valid_faces = []
            for face in faces:
                quality = FaceQualityAssessor.assess_quality(image, face)
                if quality > 0.5:
                    valid_faces.append(face)

            # Blur faces
            blurred_image = image.copy()
            for face in valid_faces:
                blurred_image = self.blurring_engine.blur_face(
                    blurred_image,
                    face,
                    strategy=self.blur_strategy,
                    blur_strength=self.blur_strength,
                )

            result.num_faces_blurred = sum(1 for f in valid_faces if f.blurred)
            result.success = True

            # Save blurred image
            output_path = self.output_dir / "blurred" / image_path.name
            cv2.imwrite(str(output_path), blurred_image)

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"Error processing {image_path.name}: {e}")

        result.processing_time = time.time() - start_time
        return result

    def run(self) -> Dict[str, Any]:
        """Process all images sequentially."""
        # Find images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = sorted(
            [
                f
                for f in self.input_dir.iterdir()
                if f.suffix.lower() in image_extensions
            ]
        )

        if not image_files:
            self.logger.error(f"No images found in {self.input_dir}")
            return {}

        self.logger.info(f"Found {len(image_files)} images to process")
        self.stats["total_images"] = len(image_files)

        # Process sequentially
        results = []
        for image_path in tqdm(image_files, desc="Processing", unit="img"):
            try:
                result = self.process_image(image_path)
                results.append(result)

                if result.success:
                    self.stats["successful"] += 1
                    self.stats["total_faces_detected"] += result.num_faces_detected
                    self.stats["total_faces_blurred"] += result.num_faces_blurred
                    self.stats["processing_time"] += result.processing_time
                else:
                    self.stats["failed"] += 1

                # Cleanup GPU memory periodically
                if self.stats["successful"] % 50 == 0:
                    gc.collect()
                    if self.device == "cuda":
                        torch.cuda.empty_cache()

            except KeyboardInterrupt:
                self.logger.info("Processing interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                self.stats["failed"] += 1

        # Save metadata
        self._save_metadata(results)
        self._print_summary()

        return self.stats

    def _save_metadata(self, results: List[FrameBlurringResult]) -> None:
        """Save processing metadata."""
        metadata = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": self.stats,
            "configuration": {
                "blur_strategy": self.blur_strategy,
                "blur_strength": self.blur_strength,
                "device": self.device,
                "use_ensemble": self.use_ensemble,
            },
            "frames": [r.to_dict() for r in results if r.success],
        }

        metadata_file = self.output_dir / "metadata" / "processing_report.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to {metadata_file}")

    def _print_summary(self) -> None:
        """Print processing summary."""
        self.logger.info("=" * 70)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Total images: {self.stats['total_images']}")
        self.logger.info(f"Successful: {self.stats['successful']}")
        self.logger.info(f"Failed: {self.stats['failed']}")

        success_rate = (
            100 * self.stats["successful"] / max(1, self.stats["total_images"])
        )
        self.logger.info(f"Success rate: {success_rate:.2f}%")

        self.logger.info(f"Total faces detected: {self.stats['total_faces_detected']}")
        self.logger.info(f"Total faces blurred: {self.stats['total_faces_blurred']}")

        avg_faces = self.stats["total_faces_detected"] / max(
            1, self.stats["successful"]
        )
        self.logger.info(f"Avg faces per image: {avg_faces:.2f}")

        total_time = self.stats["processing_time"]
        avg_time = total_time / max(1, self.stats["successful"])
        self.logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        self.logger.info(f"Avg time per image: {avg_time:.3f}s")
        self.logger.info("=" * 70)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Face Blurring for Public Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--blur_strategy",
        choices=["gaussian", "pixelate", "mosaic"],
        default="gaussian",
    )
    parser.add_argument(
        "--blur_strength",
        choices=["conservative", "moderate", "aggressive"],
        default="aggressive",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda:0",
        help="Use CPU (recommended) to avoid GPU conflicts",
    )
    parser.add_argument(
        "--disable_ensemble", action="store_true", help="Disable YOLOv8 fallback"
    )
    parser.add_argument("--disable_mediapipe", action="store_true")
    parser.add_argument("--disable_yolov8", action="store_true")

    args = parser.parse_args()

    blurrer = PrivacyPreservingFaceBlurrer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        blur_strategy=args.blur_strategy,
        blur_strength=args.blur_strength,
        use_ensemble=not args.disable_ensemble,
        device=args.device,
        use_mediapipe=not args.disable_mediapipe,
        use_yolov8=not args.disable_yolov8,
    )

    blurrer.run()


if __name__ == "__main__":
    main()
