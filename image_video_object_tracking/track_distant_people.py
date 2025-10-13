import os
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from groundingdino.util.inference import Model as GroundingDINOModel
from segment_anything_hq import SamPredictor, sam_model_registry

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
YOLO_WEIGHTS_PATH = Path("weights/yolov8l.pt")
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_WEIGHTS_PATH = Path("weights/groundingdino_swint_ogc.pth")
SAM_WEIGHTS_PATH = Path("weights/sam_hq_vit_h.pth")
SAM_MODEL_TYPE = "vit_h"

def load_yolo_sahi_model():
    print("Loading YOLOv8 model for SAHI")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type = 'yolov8',
        model_path = YOLO_WEIGHTS_PATH,
        confidence_threshold = 0.3,
        device=DEVICE,
    )
    return detection_model

def load_grounding_dino_model():
    print("Loading GroundingDINO model")
    model = GroundingDINOModel(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_WEIGHTS_PATH,
        device=DEVICE.type,
    )
    return model

def load_sam_model():
    print("Loading SAM2 model")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_WEIGHTS_PATH).to(device=DEVICE)
    predictor = SamPredictor(sam)
    return predictor

def process_frame(
    frame: np.ndarray,
    yolo_tracker: YOLO,
    sahi_model: AutoDetectionModel,
    gd_model: GroundingDINOModel,
    sam_predictor: SamPredictor
) -> np.ndarray:
    sahi_result = get_sliced_prediction(
        frame,
        sahi_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )
    detections = sv.Detections.from_sahi(sahi_result)
    person_detections = detections[detections.class_id == 0]
    tracker_results = yolo_tracker.track(source=frame, persist=True, boxes=person_detections.xyxy)
    
    if tracker_results and len(tracker_results[0].boxes.id) > 0:
        tracked_detections = sv.Detections(
            xyxy=tracker_results[0].boxes.xyxy.cpu().numpy(),
            tracker_id=tracker_results[0].boxes.id.cpu().numpy().astype(int)
        )
    else:
        return frame
    
    annotated_frame = frame.copy()
    if len(tracked_detections) > 0:
        sam_predictor.set_image(frame)
        enhanced_detections = []
        for detection_xyxy, tracker_id in zip(tracked_detections.xyxy, tracked_detections.tracker_id):
            refined_boxes, _ = gd_model.predict_with_caption(
                image=frame,
                caption="person",
                box_threshold=0.2,
                text_threshold=0.2,
                custom_box=torch.Tensor(detection_xyxy).to(DEVICE)
            )
            if len(refined_boxes) > 0:
                refined_box = refined_boxes[0].cpu().numpy()
                masks, _, _ = sam_predictor.predict(box=refined_box, multimask_output=False)
                
                det = sv.Detections(
                    xyxy = np.array([refined_box]),
                    mask = masks.squeeze(axis=0),
                    tracker_id=np.array([tracker_id])
                )
                enhanced_detections.append(det)
                
        if enhanced_detections:
            final_detections = sv.Detections.merge(detections_list=enhanced_detections)
            bounding_box_annotator = sv.BoxAnnotator(thickness=2)
            mask_annotator = sv.MaskAnnotator(opacity=0.4)
            label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.TOP_CENTER,
                text_scale=0.6,
                text_color=sv.Color.BLACK,
                # text_background_color=sv.Color.WHITE,
                text_padding=2
            )
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=final_detections)
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=final_detections)
            
            labels = [f"Person #{tracker_id}" for tracker_id in final_detections.tracker_id]
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=final_detections, labels=labels
            )
    return annotated_frame

def main(input_dir: str, output_dir: str):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    sahi_model = load_yolo_sahi_model()
    yolo_tracker = YOLO(YOLO_WEIGHTS_PATH)
    gd_model = load_grounding_dino_model()
    sam_predictor = load_sam_model()
    image_paths = sorted([p for p in Path(input_dir).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    print(f"\nFound {len(image_paths)} images. Starting processing...")
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing frame {i+1}/{len(image_paths)}: {image_path.name}")
        
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Could not read image {image_path.name}. Skipping.")
            continue
        annotated_frame = process_frame(frame, yolo_tracker, sahi_model, gd_model, sam_predictor)
        output_path = Path(output_dir) / image_path.name
        cv2.imwrite(str(output_path), annotated_frame)
        
    print(f"\nProcessing complete. Annotated frames saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track and segment distant people in equirectangular images.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing input image frames."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Path to the directory where annotated frames will be saved."
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)