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
from tqdm import tqdm

try:
    from mmpose.apis import MMPoseInferencer
    import torchreid
    from ultralytics import YOLO, SAM
    import mmcv
except ImportError as e:
    print(f"CRITICAL MISSING LIB: {e}")
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
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    # is_moving: bool
    status: str  # Moving or static
    pose_keypoints: List[Tuple[float, float]]
    segmentation_polygon: List[List[float]] = field(default_factory=list)


class MovementAnalyst:
    def __init__(self, history_len: int, cooldown_frames: int = 5):
        # Stores history: {track_id: deque([(left_ankle, right_ankle, box_height), ...])}
        # Stores normalized leg metrics
        self.history = defaultdict(lambda: deque(maxlen=history_len))
        
        # Stores how many frames to keep "Moving" status after movement stops
        self.cooldowns = defaultdict(int)
        self.cooldown_limit = cooldown_frames
        
        # self.threshold = movement_threshold
        
        # 1. Standard Deviation of Hip-Ankle distance (Vertical Swing)
        self.var_threshold = 0.01 
        # 2. Normalized distance between Left/Right Ankles (Horizontal Stride)
        # If ankles are wider than 40% of box height, likely walking
        self.stride_threshold = 0.40

    def update(
        self, track_id: int, keypoints: List[Tuple[float, float]], bbox: List[float]
    ) -> str:
        # COCO Indices: 15=Left Ankle, 16=Right Ankle
        if len(keypoints) < 17:
            return "Unknown"

        l_hip = keypoints[11]
        r_hip = keypoints[12]
        l_ankle = keypoints[15]
        r_ankle = keypoints[16]

        # LOGIC: Calculate vertical distance from Hip to Ankle
        # If this distance changes rapidly, the legs are swinging (Walking).
        # If this distance is constant, the legs are planted (Static),
        # even if the person moves across the screen due to robot motion.
        
        # Check visibility (MMPose returns 0,0 for invisible points)
        if l_hip[0] == 0 or l_ankle[0] == 0 or r_ankle[0] == 0: 
            return self._apply_cooldowns(track_id)

        # Calculate Box Height for Normalization (Scale Invariance)
        box_h = bbox[3] - bbox[1]
        if box_h <= 0:
            return "Unknown"

        # # add to history
        # self.history[track_id].append((l_ankle, r_ankle, box_h))

        # if len(self.history[track_id]) < 2:
        #     return "Analysing"

        # # calculating displacement
        # prev_l, prev_r, prev_h = self.history[track_id][0]  # oldest
        # curr_l, curr_r, curr_h = self.history[track_id][-1]  # newest

        # # Euclidean distance
        # dist_l = math.sqrt((curr_l[0] - prev_l[0]) ** 2 + (curr_l[1] - prev_l[1]) ** 2)
        # dist_r = math.sqrt((curr_r[0] - prev_r[0]) ** 2 + (curr_r[1] - prev_r[1]) ** 2)
        # avg_dist = (dist_l + dist_r) / 2.0

        # # normalize: movement as a percentage of body height
        # normalized_movement = avg_dist / box_h

        hip_center_y = (l_hip[1] + r_hip[1]) / 2.0
        ankle_center_y = (l_ankle[1] + r_ankle[1]) / 2.0

        # Normalized leg length estimate
        leg_len_norm = abs(ankle_center_y - hip_center_y) / box_h
        
        # How far apart are the feet horizontally
        ankle_spread_x = abs(l_ankle[0] - r_ankle[0])
        # Normalized by height (more stable than width for turning people)
        stride_norm = ankle_spread_x / box_h
        
        self.history[track_id].append(leg_len_norm)

        if len(self.history[track_id]) < 3:
            return "Analysing"

        variance = np.std(list(self.history[track_id]))
        
        # THE DECISION: Moving if (High Variance) OR (Wide Stride)
        is_moving_physically = (variance > self.var_threshold) or (stride_norm > self.stride_threshold)
        
        # if variance > self.threshold:
        #     return "Moving"
        # else:
        #     return "Static"
        
        if is_moving_physically:
            self.cooldowns[track_id] = self.cooldown_limit
            return "Moving"
        else:
            return self._apply_cooldowns(track_id)
        
    def _apply_cooldowns(self, track_id):
        if self.cooldowns[track_id] > 0:
            self.cooldowns[track_id] -= 1
            return "Moving"
        return "Static"


class GlobalIdentityRegistery:
    def __init__(self, cfg: DictConfig):
        self.registry: Dict[int, np.ndarray] = {}  # {global_id: feature_vector}
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
                self.registry[best_id] = (
                    0.9 * self.registry[best_id] + 0.1 * new_features
                )
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
        self.pose_inferencer = MMPoseInferencer(
            pose2d="human",
            device=self.device,
            det_model=None,  # We supply our own boxes
            det_cat_ids=[0],
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

    def get_pose_batch(
        self, frame: np.ndarray, bboxes: List[List[float]]
    ) -> List[List[Tuple[float, float]]]:
        if not bboxes:
            return []

        # Using the inferencer as a generator
        result_generator = self.pose_inferencer(frame, bboxes=bboxes, return_vis=False)
        results = next(result_generator)

        batch_keypoints = []
        frame_predictions = results["predictions"][0]  # unwrap frame

        for instance in frame_predictions:
            # pred['keypoints'] is typically a list of [x, y]
            kpts = [(float(kp[0]), float(kp[1])) for kp in instance["keypoints"]]
            batch_keypoints.append(kpts)

        return batch_keypoints

    @torch.inference_mode()
    def segment_box_prompt(
        self, image: np.ndarray, bbox: List[float]
    ) -> List[List[float]]:
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return []

        results = self.sam_model(image, bboxes=[bbox], verbose=False, retina_masks=True)
        polygons = []
        if results[0].masks is not None and len(results[0].masks.data) > 0:
            mask_data = results[0].masks.data[0].cpu().numpy()
            contours, _ = cv2.findContours(
                mask_data.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) > 300:
                    polygons.append(cnt.flatten().tolist())
        return polygons


def process_camera_stream(
    cam_config: DictConfig,
    system_config: DictConfig,
    engine: AIModelEngine,
    global_registry: GlobalIdentityRegistery,
    tqdm_position: int,
):
    cam_id = cam_config.id
    source_path = Path(cam_config.path)
    base_out = Path(system_config.system.output_dir) / f"cam_{cam_id}"
    img_out = base_out / "images"
    json_out = base_out / "json"
    img_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Cam {cam_id}] Reading from {source_path}")

    start_frame_idx = 0
    existing_jsons = sorted(list(json_out.glob("*.json")))
    if existing_jsons:
        try:
            last_file = existing_jsons[-1]
            last_idx = int(last_file.stem.split("_")[1])
            start_frame_idx = last_idx + 1
            if tqdm_position == 0:
                print(f"Resuming Cam {cam_id} from frame {start_frame_idx}")
        except:
            pass

    # Initialize Movement Analyst from Config
    movement_analyst = MovementAnalyst(
        history_len=system_config.analysis.movement.history_len,
        # threshold=system_config.analysis.movement.threshold,
    )

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        [f for f in source_path.iterdir() if f.suffix.lower() in valid_exts]
    )

    if not images:
        logger.warning(f"[Cam {cam_id}] No images found in {source_path}")
        return

    # position=tqdm_position makes them stack vertically (0=top, 1=next line, etc)
    pbar = tqdm(
        total=len(images),
        desc=f"Cam {cam_id}",
        position=tqdm_position,
        leave=False,
        initial=start_frame_idx,
    )

    for frame_idx, img_file in enumerate(images):
        if frame_idx < start_frame_idx:
            continue

        frame = cv2.imread(str(img_file))
        if frame is None:
            pbar.update(1)
            continue

        track_results = engine.det_model.track(
            frame,
            persist=True,
            tracker=system_config.tracker.type,
            conf=system_config.models.detection.conf_threshold,
            classes=system_config.models.detection.classes,
            verbose=False,
        )[0]

        frame_entities = []
        overlay = frame.copy()

        # Only proceed if we have tracks
        if track_results.boxes is not None and track_results.boxes.id is not None:
            boxes_xyxy = track_results.boxes.xyxy.cpu().numpy()
            track_ids = track_results.boxes.id.cpu().numpy()
            confs = track_results.boxes.conf.cpu().numpy()

            batch_bboxes_list = [b.tolist() for b in boxes_xyxy]
            all_poses = engine.get_pose_batch(frame, batch_bboxes_list)
            do_sam = frame_idx % engine.sam_interval == 0

            for i, track_id in enumerate(track_ids):
                x1, y1, x2, y2 = map(int, boxes_xyxy[i])
                bbox = [x1, y1, x2, y2]
                local_id = int(track_id)
                current_pose = all_poses[i] if i < len(all_poses) else []

                move_status = movement_analyst.update(local_id, current_pose, bbox)

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
                    segmentation_polygon=polygons,
                )
                frame_entities.append(asdict(entity))

                # Color Logic: Red=Moving, Blue=Static, Grey=Unknown
                if move_status == "Moving":
                    color = (0, 0, 255)
                    skel_color = (0, 0, 255) # Red Skeleton
                elif move_status == "Static":
                    color = (255, 0, 0)
                    skel_color = (0, 255, 0) # Green Skeleton indicates "Anchored"
                else:
                    color = (128, 128, 128)
                    skel_color = (128, 128, 128)

                # Draw SAM 2 Overlay (Semi-Transparent)
                if polygons:
                    for poly in polygons:
                        # Convert list to numpy array of points
                        pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                        # 1. Fill the polygon on the 'overlay' copy
                        cv2.fillPoly(overlay, [pts], color)
                        # 2. Draw border on original frame for sharpness
                        cv2.polylines(frame, [pts], True, (255, 255, 255), 1)

                # Draw INTELLIGENT SKELETON (Legs Only)
                # This helps you verify if the logic is looking at the legs
                # Draw Line from Left Hip(11) to Left Ankle(15)
                if len(current_pose) > 16:
                    # helper to get point
                    def gp(idx):
                        return (int(current_pose[idx][0]), int(current_pose[idx][1]))
                    
                    # Draw Bones
                    if current_pose[11][0] > 0 and current_pose[15][0] > 0:
                        cv2.line(frame, gp(11), gp(15), skel_color, 2) # L Hip -> L Ankle
                        
                    if current_pose[12][0] > 0 and current_pose[16][0] > 0:
                        cv2.line(frame, gp(12), gp(16), skel_color, 2) # R Hip -> R Ankle
                        
                    # Draw "Feet" to see stride
                    cv2.circle(frame, gp(15), 5, (0,255,255), -1)
                    cv2.circle(frame, gp(16), 5, (0,255,255), -1)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Header Label (Global ID + Status)
                label = f"ID:{global_id} | {move_status}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + tw, y1), color, -1)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

                # Draw Skeleton (Ankles highlighted)
                for kp_idx, kp in enumerate(current_pose):
                    # indices 15, 16 are ankles
                    k_color = (0, 255, 255) if kp_idx in [15, 16] else (0, 255, 0)
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, k_color, -1)

                # # Draw Polygon Overlay
                # if polygons:
                #     for poly in polygons:
                #         pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
                #         cv2.polylines(frame, [pts], True, color, 1)

        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        # Save Output
        with open(json_out / f"{frame_idx:06d}.json", "w") as f:
            json.dump(frame_entities, f, indent=2)

        cv2.imwrite(str(img_out / f"{frame_idx:06d}.jpg"), frame)
        pbar.update(1)

    pbar.close()
    # logger.info(f"[Cam {cam_id}] Stream Finished.")


@hydra.main(
    version_base=None, config_path="../config", config_name="multi_cam_mot.yaml"
)
def main(cfg: DictConfig):
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    # Initialize Shared Resoucres
    engine = AIModelEngine(cfg)
    global_registry = GlobalIdentityRegistery(cfg)
    start_time = time.time()

    # print(f"DEBUG: Type of cfg.cameras is {type(cfg.cameras)}")
    # print(f"DEBUG: Contents of cfg.cameras: {cfg.cameras}")

    camera_list = cfg.cameras
    if isinstance(camera_list, (dict, DictConfig)):
        print("WARNING: 'cameras' loaded as Dictionary. Converting to List...")
        # If it's a dict, we want the values (the camera objects), not the keys (strings)
        camera_list = list(camera_list.values())

    # excute camera streams in parallel
    with ThreadPoolExecutor(max_workers=cfg.system.num_workers) as executor:
        futures = []
        for i, cam_cfg in enumerate(camera_list):
            if isinstance(cam_cfg, str):
                print(
                    f"CRITICAL ERROR: Camera config is a string: '{cam_cfg}'. Check YAML indentation."
                )
                continue

            futures.append(
                executor.submit(
                    process_camera_stream,
                    cam_cfg,
                    cfg,
                    engine,
                    global_registry,
                    tqdm_position=i,
                )
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
