import sys
import cv2
import time
import torch
import json
import numpy as np
import hydra
import logging
import math
import warnings
from collections import deque, defaultdict
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import multiprocessing

try:
    from mmpose.apis import MMPoseInferencer
    import torchreid
    from ultralytics import YOLO, SAM
except ImportError as e:
    print(f"CRITICAL MISSING LIB: {e}")
    print("Run: pip install torchreid ultralytics && mim install mmpose")
    sys.exit(1)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(threadName)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PersonEntity:
    camera_id: int
    frame_id: int
    local_id: int
    global_id: int
    bbox: List[float] # [x1, y1, x2, y2]
    confidence: float
    # is_moving: bool
    status: str # Moving or static
    pose_keypoints: List[Tuple[float, float]]
    segmentation_polygon: List[List[float]] = field(default_factory=list)


class MovementAnalyst:
    def __init__(self, history_len: int, movement_threshold: float):
        # Stores history: {track_id: deque([(left_ankle, right_ankle, box_height), ...])}
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        self.threshold = movement_threshold

    def update(self, track_id: int, keypoints: List[Tuple[float, float]], bbox: List[float]) -> str:
        # COCO Indices: 15=Left Ankle, 16=Right Ankle
        if len(keypoints) < 17: 
            return "Unknown"
        
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]
        
        # Calculate Box Height for Normalization (Scale Invariance)
        box_h = bbox[3] - bbox[1]
        if box_h <= 0: 
            return "Unknown"
        
        # add to history
        self.history[track_id].append((l_ankle, r_ankle, box_h))
        
        if len(self.history[track_id]) < 2:
            return "Analysing"
        
        # calculating displacement
        prev_l, prev_r, prev_h = self.history[track_id][0] # oldest
        curr_l, curr_r, curr_h = self.history[track_id][-1] # newest
        
        # Euclidean distance
        dist_l = math.sqrt((curr_l[0] - prev_l[0])**2 + (curr_l[1] - prev_l[1])**2)
        dist_r = math.sqrt((curr_r[0] - prev_r[0])**2 + (curr_r[1] - prev_r[1])**2)
        avg_dist = (dist_l + dist_r) / 2.0
        
        # normalize: movement as a percentage of body height
        normalized_movement = avg_dist / box_h
        
        if normalized_movement > self.threshold:
            return "Moving"
        else:
            return "Static"
        

class GlobalIdentityRegistery:
    def __init__(self, cfg:DictConfig):
        self.registry: Dict[int, np.ndarray] = {} # {global_id: feature_vector}
        self.next_global_id = 1
        self.lock = Lock()
        self.threshold = cfg.models.reid.match_threshold
        
    def resolve_identity(self, new_features: np.ndarray) -> int:
        norm = np.linalg.norm(new_features)
        if norm < 1e-6:
            return -1
        new_features = new_features / norm
        
        with self.lock:
            best_id = -1
            best_scores = -1.0
            
            for gid, registered_feats in self.registry.items():
                score = np.dot(new_features, registered_feats)
                if score > best_scores:
                    best_scores = score
                    best_id = gid
                    
            if best_scores > self.threshold:
                # Match found: Update feature bank (Running Average)
                self.registry[best_id] = 0.9 * self.registry[best_id] + 0.1 * new_features
                self.registry[best_id] /= np.linalg.norm(self.registry[best_id])
                return best_id
            else:
                # No match: Create new ID
                new_id = self.next_global_id
                self.next_global_id += 1
                self.registry[new_id] = new_features
                return new_id


class AIModelEngine:
    def __init__(self, cfg: DictConfig):
        self.device = cfg.system.device
        logger.info(f"Initializing AI Engine : {self.device}")
        self.det_model = YOLO(cfg.models.detection.path)
        logger.info("Loading MMpose (RTMPose)")
        self.pose_inferencer = MMPoseInferencer(pose2d="human", device=self.device, det_model=None, # We supply our own boxes
            det_cat_ids=[0]
        )

        logger.info("Loading TorchReID (OSNet)")
        self.reid_model = torchreid.models.build_model(
            name=cfg.models.reid.name, num_classes=1000, loss="softmax", pretrained=True
        )
        self.reid_model.to(self.device)
        self.reid_model.eval()
        self.sam_model = SAM(cfg.models.segmentation.path)
        self.sam_interval = cfg.models.segmentation.run_every_n_frames

    @torch.inference_mode()
    def get_reid_features(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        # TorchReID expects specific preprocessing
        # resize -> tensor -> normalize
        crop = cv2.resize(crop, (256, 128))  # OSNet
        crop_t = torch.from_numpy(crop).permute(2, 0, 1).float().to(self.device)
        crop_t /= 255.0

        # standard ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        crop_t = (crop_t - mean) / std
        crop_t = crop_t.unsqueeze(0)

        features = self.reid_model(crop_t)
        return features.cpu().numpy().flatten()


    def get_pose_batch(self, frame: np.ndarray, bboxes: List[List[float]]) -> List[List[Tuple[float, float]]]:
        if not bboxes:
            return []
        
        result_generator = self.pose_inferencer(
            frame,
            bboxes=bboxes,
            return_vis=False
        )
        
        results = next(result_generator)
        
        batch_keypoints = []
        for pred in results['predictions']:
            # pred['keypoints'] is typically a list of [x, y]
            kpts = [(float(kp[0]), float(kp[1])) for kp in pred['keypoints']]
            batch_keypoints.append(kpts)
            
        return batch_keypoints
    
    @torch.inference_mode()
    def segment_box_prompt(self, image: np.ndarray, bbox: List[float]) -> List[List[float]]:
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return []
        
        results = self.sam_model(image, bboxes=[bbox], verbose=False, retina_masks=True)
        polygons = []
        if results[0].masks is not None:
            mask_data = results[0].masks.data[0].cpu().numpy()
            contours, _ = cv2.findContours(mask_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 300:
                    polygons.append(cnt.flatten().tolist())
        return polygons
    
def process_camera_stream(
    cam_config: DictConfig,
    system_config: DictConfig,
    engine: AIModelEngine,
    global_registry: GlobalIdentityRegistery
):
    cam_id = cam_config.id 
    source_path = Path(cam_config.path)
    output_dir = Path(system_config.output_dir) / f"cam_{cam_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[Cam {cam_id}] Reading from {source_path}")
    
    # Initialize Movement Analyst from Config
    movement_analyst = MovementAnalyst(
        history_len=system_config.analysis.movement.history_len,
        movement_threshold=system_config.analysis.movement.threshold
    )
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([f for f in source_path.iterdir() if f.suffix.lower() in valid_exts])
    
    if not images:
        logger.warning(f"[Cam {cam_id}] No images found in {source_path}")
        return
    
    for frame_idx, img_file in enumerate(images):
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue
        
        track_results = engine.det_model.track(
            frame, 
            persist=True, 
            tracker=system_config.tracker.type,
            conf=system_config.models.detection.conf_threshold,
            classes=system_config.models.detection.classes,
            verbose=system_config.system.verbose
        )[0]
        
        frame_entities = []
        
        # Only proceed if we have tracks
        if track_results.boxes is not None and track_results.boxes.id is not None:
            boxes_xyxy = track_results.boxes.xyxy.cpu().numpy()
            track_ids = track_results.boxes.id.cpu().numpy()
            confs = track_results.boxes.conf.cpu().numpy()
            
            batch_bboxes_list = [b.tolist() for b in boxes_xyxy]
            all_poses = engine.get_pose_batch(frame, batch_bboxes_list)
            
            do_sam = (frame_idx % engine.sam_interval == 0)
            
            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                bbox = [x1, y1, x2, y2]
                local_id = int(track_id)
                current_pose = all_poses[i] if i < len(all_poses) else []
                
                move_status = movement_analyst.update(local_id, current_pose, bbox)
                is_moving = (move_status == "Moving")
                
                # crop person for ReID
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                reid_crop = frame[cy1:cy2, cx1:cx2]
                
                reid_feats = engine.get_reid_features(reid_crop)
                global_id = global_registry.resolve_identity(reid_feats)
                
                polygons = []
                if do_sam:
                    polygons = engine.segment_box_prompt(frame, bbox)
                    
                entity = PersonEntity(
                    camera_id=cam_id,
                    frame_id=frame_idx,
                    local_id=local_id,
                    global_id=global_id,
                    bbox=bbox,
                    confidence=float(confs[i]),
                    status=move_status,
                    pose_keypoints=current_pose,
                    segmentation_polygon=polygons
                )
                frame_entities.append(asdict(entity))
                
                # Color Logic: Red=Moving, Blue=Static, Grey=Unknown
                if move_status == "Moving": 
                    color = (0, 0, 255) 
                elif move_status == "Static": 
                    color = (255, 0, 0)
                else: 
                    color = (128, 128, 128)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Header Label (Global ID + Status)
                label = f"ID:{global_id} | {move_status}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw Skeleton (Ankles highlighted)
                for kp_idx, kp in enumerate(current_pose):
                    # indices 15, 16 are ankles
                    k_color = (0, 255, 255) if kp_idx in [15, 16] else (0, 255, 0)
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, k_color, -1)
                    
                
                # Draw Polygon Overlay
                if polygons:
                    for poly in polygons:
                        pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                        cv2.polylines(frame, [pts], True, color, 1)
                        
        # Save Output
        json_path = output_dir / f"{frame_idx:06d}.json"
        with open(json_path, 'w') as f:
            json.dump(frame_entities, f, indent=2)
        
        cv2.imwrite(str(output_dir / f"{frame_idx:06d}.jpg"), frame)
    
    logger.info(f"[Cam {cam_id}] Stream Finished.")
    
    
@hydra.main(version_base=None, config_path="../config", config_name="multi_cam_mot.yaml")
def main(cfg: DictConfig):
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Initialize Shared Resoucres
    engine = AIModelEngine(cfg)
    global_registry = GlobalIdentityRegistery(cfg)
    start_time = time.time()
    
    # excute camera streams in parallel
    with ThreadPoolExecutor(max_workers=cfg.system.num_workers) as executor:
        futures = []
        for cam_cfg in cfg.cameras:
            futures.append(
                executor.submit(process_camera_stream, cam_cfg, cfg, engine, global_registry)
            )
            
            # wait for all cameras to finish
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Pipeline Error: {e}", exc_info=True)
                    
    duration = time.time() - start_time
    print(f"\nProcessing Complete. Total Time: {duration:.2f}s")
    print(f"Output saved to: {cfg.system.output_dir}")

if __name__ == "__main__":
    main()