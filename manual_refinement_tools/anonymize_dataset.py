import os
import cv2
import glob
import logging
import argparse
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from mutliprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("anonymization.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FaceAnonymizer:
    def __init__(
        self, detection_threshold: float = 0.5, device: str = "cuda:0"
    ) -> None:
        # detection_threshold (float): Confidence threshold (0.0 to 1.0). Lower = more faces detected but more false positives.
        self.app = FaceAnalysis(allowed_modules=["detection"])
        self.app.prepare(
            ctx_id=0 if device == "cuda:0" else -1, det_thresh=detection_threshold
        )
        self.threshold = detection_threshold

    def get_dynamic_kernel(self, w: int, h: int) -> Tuple[int, int]:
        """
        Calculates a blur kernel size relative to the face dimensions.
        Ensures the kernel is always odd (required by GaussianBlur).
        """
        # A factor of 0.15 (15%) provides a strong blur without destroying context
        kernel_width = int(min(w, h) * 0.15)
        kernel_height = int(min(w, h) * 0.15)

        # ensuring that the kernel is odd
        if kernel_height % 2 == 0:
            kernel_height += 1
        if kernel_width % 2 == 0:
            kernel_width += 1

        # minimum safe kernel size
        return (max(3, kernel_width), max(3, kernel_height))

    def blur_faces(self, img: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detects faces in the image and applies a Gaussian blur.

        Returns:
            processed_img (np.ndarray): The image with blurred faces.
            face_count (int): Number of faces blurred.
        """

        # InsightFace expects BGR (OpenCV default), no conversion needed for detection
        faces = self.app.get(img)

        if not faces:
            return img, 0

        processed_img = img.copy()

        for face in faces:
            # Bounding box is [x_min, y_min, x_max, y_max]
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Clip coordinates to image boundaries (safety check)
            h_img, w_img = processed_img.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w_img, x2)
            y2 = min(h_img, y2)

            # Face dimensions
            w_face = x2 - x1
            h_face = y2 - y1

            if w_face <= 0 or h_face <= 0:
                continue

            roi = processed_img[y1:y2, x1:x2]

            # Calculate dynamic kernel based on this specific face size
            kernel_size = self.get_dynamic_kernel(w_face, h_face)

            # Apply Gaussian Blur
            # SigmaX=0 lets OpenCV calculate sigma from kernel size automatically
            blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)

            # Place blurred ROI back into image
            processed_img[y1:y2, x1:x2] = blurred_roi

        return processed_img, len(faces)


# Global initializer for worker processes so they don't reload the model every time
detect_model = None


def init_worker(threshold, device):
    global detect_model
    detect_model = FaceAnonymizer(detection_threshold=threshold, device=device)


def process_single_image(args):
    input_path, output_dir = args
    global detect_model

    try:
        filename = os.path.basename(input_path)
        save_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)
        if img is None:
            return f"Error: Could not read {filename}"

        processed_img, face_count = detect_model.blur_faces(img)

        cv2.imwrite(save_path, processed_img)
        return face_count

    except Exception as e:
        return f"Exception on {input_path}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="High-Precision Face Anonymization for Datasets"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input directory containing images",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Detection confidence (0.4 recommended for public datasets to ensure recall)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use. Note: Multiprocessing on CUDA is complex; CPU is recommended for stability unless you use batch inference.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of CPU workers",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_path, ext)))
        image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))

    image_files = sorted(list(set(image_files)))
    if not image_files:
        logger.error("No images found in input directory!")
        return

    logger.info(
        f"Found {len(image_files)} images. Starting processing with {args.workers} workers..."
    )
    logger.info(
        f"Model: InsightFace (SCRFD) | Threshold: {args.threshold} | Device: {args.device}"
    )

    # We prepare arguments for map
    task_args = [(f, str(output_path)) for f in image_files]

    total_faces = 0
    errors = []

    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.threshold, args.device),
    ) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_image, task_args),
                total=len(image_files),
                unit="img",
            )
        )

    for res in results:
        if isinstance(res, int):
            total_faces += res
        else:
            errors.append(res)

    logger.info("--- Processing Complete ---")
    logger.info(f"Successfully processed images: {len(image_files) - len(errors)}")
    logger.info(f"Total faces blurred: {total_faces}")

    if errors:
        logger.error(f"Encountered {len(errors)} errors:")
        for err in errors[:10]:  # Print first 10 errors
            logger.error(err)
        if len(errors) > 10:
            logger.error("...and more.")


if __name__ == "__main__":
    main()
