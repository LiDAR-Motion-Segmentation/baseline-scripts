import json
import os
import numpy as np
from shapely.geometry import Polygon

# --- Configuration ---
GT_DIR = "./path/to/raw_gt_labels"          # Folder with your extracted GT JSONs
PRED_DIR = "./path/to/exported_annotations" # Folder with your SUSTech exported JSONs
IOU_THRESHOLD = 0.5                         # Standard threshold for cars (use 0.25 for pedestrians)

def get_bev_polygon(x, y, w, l, yaw):
    """Calculates the 4 corners of the bounding box in Bird's Eye View (BEV)"""
    # Create unrotated corners
    corners = np.array([
        [-l/2, -w/2],
        [ l/2, -w/2],
        [ l/2,  w/2],
        [-l/2,  w/2]
    ])
    
    # Rotation matrix for Yaw
    rot_mat = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw),  np.cos(yaw)]
    ])
    
    # Rotate and translate
    corners_rotated = np.dot(corners, rot_mat.T) + np.array([x, y])
    return Polygon(corners_rotated)

def compute_3d_iou(box1, box2):
    """Computes 3D IoU by multiplying BEV IoU by Z-axis overlap"""
    # Unpack Box 1
    x1, y1, z1 = box1['position']['x'], box1['position']['y'], box1['position']['z']
    w1, l1, h1 = box1['scale']['x'], box1['scale']['y'], box1['scale']['z']
    yaw1 = box1['rotation']['z']
    
    # Unpack Box 2
    x2, y2, z2 = box2['position']['x'], box2['position']['y'], box2['position']['z']
    w2, l2, h2 = box2['scale']['x'], box2['scale']['y'], box2['scale']['z']
    yaw2 = box2['rotation']['z']
    
    # 1. BEV Intersection (Shapely)
    poly1 = get_bev_polygon(x1, y1, w1, l1, yaw1)
    poly2 = get_bev_polygon(x2, y2, w2, l2, yaw2)
    
    if not poly1.intersects(poly2):
        return 0.0
        
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.area + poly2.area - inter_area
    
    # 2. Height (Z) Intersection
    z1_min, z1_max = z1 - h1/2, z1 + h1/2
    z2_min, z2_max = z2 - h2/2, z2 + h2/2
    
    inter_h = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    union_h = max(z1_max, z2_max) - min(z1_min, z2_min)
    
    if inter_h == 0:
        return 0.0
        
    # 3. Final 3D IoU
    inter_vol = inter_area * inter_h
    vol1 = poly1.area * h1
    vol2 = poly2.area * h2
    union_vol = vol1 + vol2 - inter_vol
    
    return inter_vol / union_vol if union_vol > 0 else 0.0

def evaluate_frame(gt_boxes, pred_boxes, threshold=0.5):
    """Matches predictions to GT boxes to calculate TP, FP, FN"""
    tp = 0
    matched_gt = set()
    matched_pred = set()
    
    # Sort predictions by confidence if available, else standard loop
    for p_idx, p_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for g_idx, g_box in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue # Already matched this GT
                
            iou = compute_3d_iou(p_box['psr'], g_box['psr'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
                
        if best_iou >= threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    
    return tp, fp, fn

def main():
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Find all JSONs in the prediction directory
    files = [f for f in os.listdir(PRED_DIR) if f.endswith('.json')]
    
    print(f"Evaluating {len(files)} frames at IoU Threshold: {IOU_THRESHOLD}...\n")
    
    for file in sorted(files):
        pred_path = os.path.join(PRED_DIR, file)
        gt_path = os.path.join(GT_DIR, file)
        
        if not os.path.exists(gt_path):
            print(f"Skipping {file}: No corresponding GT found.")
            continue
            
        with open(pred_path, 'r') as f:
            pred_boxes = json.load(f)
        with open(gt_path, 'r') as f:
            gt_boxes = json.load(f)
            
        # Optional: Filter by specific class if you only want to benchmark cars
        # pred_boxes = [b for b in pred_boxes if 'car' in b['obj_type']]
        # gt_boxes = [b for b in gt_boxes if 'car' in b['obj_type']]
            
        tp, fp, fn = evaluate_frame(gt_boxes, pred_boxes, IOU_THRESHOLD)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"Frame {file}: TP={tp}, FP={fp}, FN={fn}")
        
    # Calculate Final Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*30)
    print("      FINAL BENCHMARK      ")
    print("="*30)
    print(f"Total True Positives (TP) : {total_tp}")
    print(f"Total False Positives (FP): {total_fp}  <- (Ghosts/Bad Annotations)")
    print(f"Total False Negatives (FN): {total_fn}  <- (Missed Objects)")
    print("-" * 30)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1_score:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()