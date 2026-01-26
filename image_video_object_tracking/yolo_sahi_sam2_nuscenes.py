"""
yolo_sahi_sam2_nuscenes.py

Pipeline:
1. SAHI + YOLO: Slices image to detect small/far objects.
2. SAM 2: Uses SAHI boxes as prompts for segmentation.
3. Visualization: Saves results with masks and boxes.

Usage:
    python yolo_sahi_sam2_nuscenes.py --root /data/sets/nuscenes --output ./sahi_results --sam2_ckpt sam2_hiera_large.pt
"""

import os
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

# NuScenes
from nuscenes.nuscenes import NuScenes

# SAHI
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# SAM 2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
# SAHI Slicing Config
SLICE_HEIGHT = 512
SLICE_WIDTH = 512
OVERLAP_HEIGHT_RATIO = 0.2
OVERLAP_WIDTH_RATIO = 0.2

# Class Mapping (COCO -> NuScenes Interest)
# 0: Person, 2: Car, 5: Bus, 7: Truck
TARGET_CLASSES = [0, 2, 5, 7]

COLORS = {
    0: (0, 0, 255),  # Person: Red
    2: (0, 255, 0),  # Car: Green
    5: (255, 255, 0),  # Bus: Cyan
    7: (255, 0, 255),  # Truck: Magenta
    "mask": (255, 100, 0),
}


class Pipeline:
    def __init__(self, args):
        self.output_dir = args.output
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Initialize NuScenes
        print(f"[Info] Loading NuScenes from {args.root}...")
        version = "v1.0-mini" if "mini" in args.root else "v1.0-trainval"
        try:
            self.nusc = NuScenes(version=version, dataroot=args.root, verbose=False)
        except Exception as e:
            print(f"[Error] Failed to load NuScenes: {e}")
            exit(1)

        # 2. Initialize SAHI Model (Wraps YOLO)
        print(f"[Info] Loading SAHI model with {args.yolo_model}...")
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=args.yolo_model,
            confidence_threshold=0.35,  # Lower conf allowed because slicing helps precision
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # 3. Initialize SAM 2
        print(f"[Info] Loading SAM 2 model ({args.sam2_config})...")
        sam2_model = build_sam2(args.sam2_config, args.sam2_ckpt, device="cuda")
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def process_image(self, img_path, save_name):
        # A. Read Image
        image = cv2.imread(img_path)
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # B. Run SAHI (Sliced Inference)
        # This creates multiple crops, runs YOLO on each, and merges results (NMS)
        sahi_result = get_sliced_prediction(
            image_rgb,
            self.detection_model,
            slice_height=SLICE_HEIGHT,
            slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        )

        # Extract Boxes for SAM 2
        # SAHI ObjectPrediction.bbox is usually [minx, miny, maxx, maxy] (shifted)
        boxes = []
        class_ids = []

        for obj in sahi_result.object_prediction_list:
            cls_id = obj.category.id
            if cls_id in TARGET_CLASSES:
                # SAHI returns bbox as [minx, miny, maxx, maxy]
                boxes.append(obj.bbox.to_xyxy())
                class_ids.append(cls_id)

        boxes = np.array(boxes)

        if len(boxes) == 0:
            cv2.imwrite(os.path.join(self.output_dir, save_name), image)
            return

        # C. Run SAM 2 (Prompted by SAHI boxes)
        self.sam2_predictor.set_image(image_rgb)

        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )

        # D. Visualize
        vis_img = image.copy()

        # 1. Masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Create a single blended overlay for speed
        overlay = np.zeros_like(vis_img, dtype=np.uint8)
        alpha_mask = np.zeros(vis_img.shape[:2], dtype=bool)

        for i, mask in enumerate(masks):
            color = COLORS.get(class_ids[i], COLORS["mask"])
            bin_mask = mask > 0.5
            overlay[bin_mask] = color
            alpha_mask |= bin_mask  # Accumulate mask area

        # Apply Blend
        vis_img[alpha_mask] = cv2.addWeighted(
            vis_img[alpha_mask], 0.5, overlay[alpha_mask], 0.5, 0
        )[0]

        # 2. Boxes & Labels
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = class_ids[i]
            color = COLORS.get(cls_id, (255, 255, 255))

            # Label
            label_name = {0: "Person", 2: "Car", 5: "Bus", 7: "Truck"}.get(
                cls_id, f"ID {cls_id}"
            )

            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis_img,
                label_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        # E. Save
        cv2.imwrite(os.path.join(self.output_dir, save_name), vis_img)

    def run(self):
        print("[Info] Starting SAHI processing loop...")

        for scene in tqdm(self.nusc.scene, desc="Scenes"):
            token = scene["first_sample_token"]

            while token != "":
                sample = self.nusc.get("sample", token)

                # Process all 6 cameras
                cams = [
                    "CAM_FRONT",
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                    "CAM_BACK_RIGHT",
                ]

                for cam in cams:
                    cam_data = self.nusc.get("sample_data", sample["data"][cam])
                    filename = cam_data["filename"]
                    full_path = os.path.join(self.nusc.dataroot, filename)

                    base_name = os.path.basename(filename)
                    save_name = f"{scene['name']}_{cam}_{base_name}"

                    self.process_image(full_path, save_name)

                token = sample["next"]

        print(f"[Success] Done. Results in {self.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI + YOLO + SAM2 on NuScenes")

    parser.add_argument("--root", type=str, required=True, help="NuScenes root path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")

    parser.add_argument(
        "--yolo_model", type=str, default="yolo11l.pt", help="YOLO model path"
    )
    parser.add_argument(
        "--sam2_ckpt", type=str, required=True, help="SAM 2 checkpoint (.pt)"
    )
    parser.add_argument(
        "--sam2_config", type=str, default="sam2_hiera_l.yaml", help="SAM 2 config"
    )

    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.run()
