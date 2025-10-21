from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import supervision as sv
import math
import yaml
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from segment_anything_hq import SamPredictor, sam_model_registry
import argparse
import time

def get_center_point(bbox: np.ndarray) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

def track_and_annotate(data: str | Path,
                       config: dict,
                       output_dir: str | Path | None = None):
    print("initializing models")
    data_path = Path(data)
    
    paths_cfg = config['paths']
    models_cfg = config['models']
    sahi_cfg = config['sahi_params']
    detection_cfg = config['detection_params']
    tracking_cfg = config['tracking_params']
    device = torch.device(models_cfg['device'] or ("cuda" if torch.cuda.is_available() else "cpu"))
    
    if not output_dir:
        output_dir = data_path.parent / f"{data_path.stem}_advanced_annotations"
    output_dir = Path(output_dir)
    labels_output_dir = output_dir / "labels_txt"
    json_output_dir = output_dir / "labels_json"
    vis_output_dir = output_dir / "visualizations"
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8', 
        model_path=paths_cfg['yolo_model'], 
        confidence_threshold=detection_cfg['confidence_threshold'], 
        device=device
    )
    
    # using CPU version as of now
    sam = sam_model_registry[models_cfg['sam_model_type']]()
    state_dict = torch.load(paths_cfg['sam_checkpoint'], map_location='cpu')
    sam.load_state_dict(state_dict)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    mask_annotator = sv.MaskAnnotator(opacity=0.4)
    label_annotator = sv.LabelAnnotator(text_scale=0.6, 
                                        text_color=sv.Color.BLACK)
    
    tracker_history:dict = {}
    image_paths = sorted([p for p in data_path.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    
    for i, image_path in enumerate(image_paths):
        vis_fn = vis_output_dir / image_path.name
        json_fn = json_output_dir / f"{image_path.stem}.json"
        label_fn = labels_output_dir / f"{image_path.stem}.txt"
        if vis_fn.exists() and json_fn.exists() and label_fn.exists():
            print(f"skipping already processed frame : {image_path.name}")
            continue
        
        print(f"Processing frame {i+1}/{len(image_paths)}: {image_path.name}")
        
        try: 
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            
            sahi_result = get_sliced_prediction(
                frame,        
                detection_model,
                slice_height=sahi_cfg['slice_height'],
                slice_width=sahi_cfg['slice_width'],
                overlap_height_ratio=sahi_cfg['overlap_ratio'],
                overlap_width_ratio=sahi_cfg['overlap_ratio'])
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
            detections = detections.with_nms(threshold=detection_cfg['nms_threshold'])
            
            tracked_detections = tracker.update_with_detections(detections)
            tracked_detections = tracked_detections[tracked_detections.tracker_id != None]
            
            if len(tracked_detections) == 0:
                continue
            
            sam_predictor.set_image(frame)
            masks_tensor, _, _ = sam_predictor.predict_torch(
                point_coords=None, point_labels=None,
                boxes=torch.tensor(tracked_detections.xyxy).to(device),
                multimask_output=False
            )
            tracked_detections.mask = masks_tensor.cpu().numpy()
            
            json_frame_data = []
            custom_labels = []
            
            with open(labels_output_dir / f"{image_path.stem}.txt", "w", encoding="utf-8") as f_text:
                for detection_idx in range(len(tracked_detections)):
                    bbox = tracked_detections.xyxy[detection_idx]
                    track_id = tracked_detections.tracker_id[detection_idx]
                    mask = tracked_detections.mask[detection_idx]
                    class_id = tracked_detections.class_id[detection_idx]
                    center = get_center_point(bbox)
                    obj_status = "people.moving"
                    
                    # Squeeze the mask to remove the singleton dimension (from 1xHxW to HxW)
                    squeezed_mask = mask.squeeze()
                    
                    if track_id in tracker_history:
                        last_center, static_frames = tracker_history[track_id]
                        distance = math.dist(center, last_center)
                        if distance < tracking_cfg['movement_threshold_pixels']:
                            static_frames += 1
                        else:
                            static_frames = 0
                            
                        if static_frames >= tracking_cfg['static_frame_count_threshold']:
                            obj_status = "people.static"
                        tracker_history[track_id] = (center, static_frames)
                    else:
                        tracker_history[track_id] = (center, 0)
                    
                    custom_labels.append(f"#{track_id} {obj_status.split('.')[1]}")
                    
                    box_width = bbox[2] - bbox[0]
                    box_height = bbox[3] - bbox[1]
                    json_obj = {
                        "obj_id": str(track_id),
                        "obj_type": obj_status,
                        "psr": {
                            "position": {"x": center[0], "y": center[1], "z": -0.5},
                            "rotation": {"x": 0, "y": 0, "z": 1},
                            "scale": {"x": box_width, "y": box_height, "z": (box_height * 2.5)}
                        }
                    }
                    json_frame_data.append(json_obj)
                    
                    if np.any(squeezed_mask):
                        polygons = sv.mask_to_polygons(squeezed_mask)
                        if polygons:
                            segment = polygons[0] / np.array([frame.shape[1], frame.shape[0]])
                            segment_str = " ".join(map(str, segment.flatten()))
                            f_text.write(f"{class_id} {segment_str}\n")
                        
            with open(json_output_dir / f"{image_path.stem}.json", "w", encoding="utf-8") as f_json:
                json.dump(json_frame_data, f_json, indent=2)    
                
            # experimenting a mask reshaping functionality    
            if len(tracked_detections) > 0:
                H, W, _ = frame.shape
                scale = 1024/ max(H, W)
                frame_for_sam = cv2.resize(frame, (int(W * scale), int(H * scale)))
                sam_predictor.set_image(frame_for_sam)
                
                scaled_boxes = tracked_detections.xyxy * scale
                masks_tensor, _, _ = sam_predictor.predict_torch(point_coords= None,
                                                                point_labels= None,
                                                                boxes= torch.tensor(scaled_boxes).to(device),
                                                                multimask_output= False)
                masks_np = masks_tensor.cpu().numpy()
                num_detections = len(tracked_detections)
                final_masks = np.zeros((num_detections, H, W), dtype=bool)
                
                for idx, scaled_mask in enumerate(masks_np):
                    mask_2d = scaled_mask.squeeze()
                    resize_mask = cv2.resize(
                        mask_2d.astype(np.uint8),
                        (W, H),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    final_masks[idx] = resize_mask
                    
                tracked_detections.mask = final_masks
            
            annotated_frame = frame.copy()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=tracked_detections, labels=custom_labels)
            cv2.imwrite(str(vis_output_dir / image_path.name), annotated_frame)
            
            print(f"Processing complete. Results saved in: {output_dir}")
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory on frame {image_path.name}, skipping...")
                torch.cuda.empty_cache()
                time.sleep(10)  # small pause to let VRAM clear
                continue
            else:
                print(f"RuntimeError on {image_path.name}: {e}")
                continue
            
        except Exception as e:
            print(f"Exception processing {image_path.name}: {e}")
            continue
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced annotation and tracking with YOLO, SAHI, SAM, ByteTrack")
    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="Path to the directory containing input image frames."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Path to the directory where all results will be saved. (Optional)"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yml", 
        help="Path to the configuration YAML file. (Optional)"
    )
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()

    track_and_annotate(data=args.data, config=config, output_dir=args.output_dir)