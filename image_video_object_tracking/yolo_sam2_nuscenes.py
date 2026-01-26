"""
yolo_sam2_organized.py

Pipeline:
1. Standard YOLO (No SAHI) for speed.
2. SAM 2 for Segmentation.
3. Output sorted into subfolders per camera (e.g., output/CAM_FRONT/).
4. Fixed colors for Persons (Red), Cars (Green), etc.

Usage:
    python yolo_sam2_organized.py \
      --root /data/sets/nuscenes \
      --output ./organized_results \
      --yolo_model yolo11l.pt \
      --sam2_ckpt sam2_hiera_large.pt \
      --sam2_config sam2_hiera_l.yaml
"""

import os
import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm

# NuScenes
from nuscenes.nuscenes import NuScenes

# YOLO
from ultralytics import YOLO

# SAM 2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION ---
# COCO Indices: 0=Person, 2=Car, 5=Bus, 7=Truck
TARGET_CLASSES = [0, 2, 5, 7]

# Visualization Colors (BGR Format: Blue, Green, Red)
CLASS_COLORS = {
    0: (0, 0, 255),  # Person: Red
    2: (0, 255, 0),  # Car: Green
    5: (255, 255, 0),  # Bus: Cyan
    7: (255, 0, 255),  # Truck: Magenta
}
DEFAULT_COLOR = (255, 255, 255)  # White fallback


class Pipeline:
    def __init__(self, args):
        self.output_root = args.output

        # 1. Initialize NuScenes
        print(f"[Info] Loading NuScenes from {args.root}...")
        version = "v1.0-mini" if "mini" in args.root else "v1.0-trainval"
        try:
            self.nusc = NuScenes(version=version, dataroot=args.root, verbose=False)
        except Exception as e:
            print(f"[Error] Failed to load NuScenes: {e}")
            exit(1)

        # 2. Initialize YOLO
        print(f"[Info] Loading YOLO ({args.yolo_model})...")
        self.yolo = YOLO(args.yolo_model)

        # 3. Initialize SAM 2
        print(f"[Info] Loading SAM 2 ({args.sam2_config})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(args.sam2_config, args.sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def process_image(self, img_path, cam_name, filename):
        # A. Prepare Output Directory
        # e.g., ./organized_results/CAM_FRONT/
        cam_dir = os.path.join(self.output_root, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        save_path = os.path.join(cam_dir, filename)

        # B. Read Image
        image = cv2.imread(img_path)
        if image is None:
            return
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # C. Run YOLO
        # conf=0.45 filters out weak detections
        results = self.yolo(
            image_rgb, classes=TARGET_CLASSES, conf=0.45, verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        if len(boxes) == 0:
            # Save original image if no objects found
            cv2.imwrite(save_path, image)
            return

        # D. Run SAM 2
        self.sam2_predictor.set_image(image_rgb)

        # Batch prediction for all boxes
        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )

        # E. Visualize
        vis_img = image.copy()

        # Handle mask shape (N, 1, H, W) -> (N, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # 1. Draw Masks (Semi-transparent)
        overlay = np.zeros_like(vis_img, dtype=np.uint8)
        alpha_mask = np.zeros(vis_img.shape[:2], dtype=bool)

        for i, mask in enumerate(masks):
            cls_id = class_ids[i]
            color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

            # Add mask to overlay
            bin_mask = mask > 0.5
            overlay[bin_mask] = color
            alpha_mask |= bin_mask

        # Blend overlay with original image
        vis_img[alpha_mask] = cv2.addWeighted(
            vis_img[alpha_mask],
            0.6,  # Image weight
            overlay[alpha_mask],
            0.4,  # Mask weight
            0,
        )[0]

        # 2. Draw Boxes & Labels
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cls_id = class_ids[i]
            color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)

            # Define Label Text
            label_map = {0: "Person", 2: "Car", 5: "Bus", 7: "Truck"}
            label_text = label_map.get(cls_id, f"ID {cls_id}")

            # Draw Box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)

            # Draw Label Background & Text
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(
                vis_img,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # F. Save
        cv2.imwrite(save_path, vis_img)

    def run(self):
        print(f"[Info] Processing started. Output will be in: {self.output_root}")

        for scene in tqdm(self.nusc.scene, desc="Processing Scenes"):
            token = scene["first_sample_token"]

            while token != "":
                sample = self.nusc.get("sample", token)

                # Iterate all 6 cameras
                cams = [
                    "CAM_FRONT",
                    "CAM_FRONT_LEFT",
                    "CAM_FRONT_RIGHT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                    "CAM_BACK_RIGHT",
                ]

                for cam_name in cams:
                    cam_token = sample["data"][cam_name]
                    cam_data = self.nusc.get("sample_data", cam_token)

                    # Get full path
                    full_path = os.path.join(self.nusc.dataroot, cam_data["filename"])

                    # Create a readable filename (Scene + Timestamp)
                    # e.g. scene-0061_1532402927612460.jpg
                    timestamp = cam_data["timestamp"]
                    filename = f"{scene['name']}_{timestamp}.jpg"

                    self.process_image(full_path, cam_name, filename)

                # Next sample
                token = sample["next"]

        print("[Success] Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + SAM2 Organized Output")

    parser.add_argument("--root", type=str, required=True, help="NuScenes root path")
    parser.add_argument(
        "--output", type=str, required=True, help="Output root directory"
    )

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
