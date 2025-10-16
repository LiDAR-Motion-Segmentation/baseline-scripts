import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import supervision as sv
import gc
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from segment_anything_hq import SamPredictor, sam_model_registry

YOLO_WEIGHTS_PATH = "/home/soumoroy/baseline-scripts/weights/yolov8l.pt"
SAM_CHECKPOINT_PATH = "/home/soumoroy/baseline-scripts/weights/sam_hq_vit_b.pth"
SAM_MODEL_TYPE = "vit_b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def initialize_models():
    print("initializing models")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=YOLO_WEIGHTS_PATH,
        confidence_threshold=0.3,
        device=DEVICE,
    )
    sam = sam_model_registry[SAM_MODEL_TYPE]()
    state_dict = torch.load(SAM_CHECKPOINT_PATH, map_location='cpu')
    sam.load_state_dict(state_dict)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    print("models have been sucessfully initialized")
    return detection_model, sam_predictor

def main(input_dir: str, output_dir: str):
    sahi_model, sam_predictor = initialize_models()
    output_path_obj = Path(output_dir)
    output_path_obj.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in Path(input_dir).glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not image_paths:
        print(f"No images found in directory: {input_dir}")
        return
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, 
                                        text_color=sv.Color.BLACK) 
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing frame {i+1}/{len(image_paths)}: {image_path.name}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Could not read image {image_path.name}. Skipping.")
            continue
        
        sahi_result = get_sliced_prediction(
            frame, 
            sahi_model, 
            slice_height=512, 
            slice_width=512,
            overlap_height_ratio=0.2, 
            overlap_width_ratio=0.2,
        )
        xyxy_list = []
        confidence_list = []
        class_id_list = []
        
        if sahi_result.object_prediction_list:
            for pred in sahi_result.object_prediction_list:
                xyxy_list.append(pred.bbox.to_xyxy())
                confidence_list.append(pred.score.value)
                class_id_list.append(pred.category.id)
            
        detections = sv.Detections(
            xyxy=np.array(xyxy_list),
            confidence=np.array(confidence_list),
            class_id=np.array(class_id_list).astype(int)   
        )
        detections = detections[detections.class_id == 0]
        detections = detections.with_nmm(threshold=0.5)

        if len(detections.xyxy) == 0:
            cv2.imwrite(str(output_path_obj / image_path.name), frame)
            continue
        
        tracked_detections = tracker.update_with_detections(detections)
        if len(tracked_detections) > 0:
            H,W, _ = frame.shape
            frame_for_sam = cv2.resize(frame, (int(W), int(H)))
            sam_predictor.set_image(frame_for_sam)
            
            masks_tensor, _, _ = sam_predictor.predict_torch(
                point_coords=None, 
                point_labels=None,
                boxes=torch.tensor(tracked_detections.xyxy).to(DEVICE),
                multimask_output=False,
            )
            
            masks_np = masks_tensor.cpu().numpy()
            final_masks = []
            for mask in masks_np:
                resized_mask = cv2.resize(mask[0].astype(np.uint8), (W, H)).astype(bool)
                final_masks.append(resized_mask)
                
            tracked_detections.mask = np.array(final_masks)
            
            annotated_frame = frame.copy()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            labels = [f"#{tracker_id}" for tracker_id in tracked_detections.tracker_id]
            annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=tracked_detections,labels=labels)
            
            cv2.imwrite(str(output_path_obj / image_path.name), annotated_frame)
        else:
            cv2.imwrite(str(output_path_obj / image_path.name), frame)
            
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Processing complete. Annotated frames saved to: {output_dir}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO+SAHI+ByteTrack+SAM Tracking on Image Frames")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory of input frames.")
    parser.add_argument("--output_dir", type=str, default="output_frames", help="Path to save the output frames.")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)