# python evaluate_points.py \
#   --pcd_dir /home/soumoroy/Downloads/annotations/aug5_sustech/nuscenes_v2/scene-0103/lidar \
#   --gt_dir /home/soumoroy/Downloads/annotations/aug5_sustech/nuscenes_v2/scene-0103/label \
#   --pred_dir /path/to/your/exported/msalt_annotations \
#   --num_frames 10

import os
import json
import argparse
import numpy as np
import open3d as o3d

def get_points_in_box(points, box):
    """
    Counts how many LiDAR points fall inside a 3D bounding box.
    Uses the Translate -> Rotate -> AABB Filter method for extreme speed.
    """
    
    # extract box properties
    cx, cy, cz = box['position']['x'], box['position']['y'], box['position']['z']
    w, l, h = box['scale']['x'], box['scale']['y'], box['scale']['z']
    yaw = box['rotation']['z']
    
    # translate points (move the box to 0,0,0)
    translated_points = points - np.array([cx, cy, cz])
    
    # rotate points (Apply Inverse Yaw to straighten the box)
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    
    # 3x3 rotation matrix around the z axis
    rot_mat = np.array([
        [cos_y, -sin_y, 0],
        [sin_y,  cos_y, 0],
        [    0,      0, 1]
    ])
    
    # apply rotation
    rotated_points = np.dot(translated_points, rot_mat.T)
    
    # AABB Filter (the fast bounding box check)
    mask_x = np.abs(rotated_points[:, 0]) <= (w / 2.0)
    mask_y = np.abs(rotated_points[:, 1]) <= (l / 2.0)
    mask_z = np.abs(rotated_points[:, 2]) <= (h / 2.0)
    
    # Combine masks
    inside_box_mask = mask_x & mask_y & mask_z

    return np.sum(inside_box_mask)

def main(pcd_dir, gt_dir, pred_dir, num_frames):
    # dictionary to hold class-wise metrics
    class_metrics = {   }
    
    # global metrics
    total_gt_points = 0
    total_pred_points = 0
    total_absolute_error = 0
    total_objects_evaluated = 0
    
    if not os.path.exists(pred_dir):
        print(f"[ERROR] Prediction directory not found: {pred_dir}")
        return
    
    files = [f for f in os.listdir(pred_dir) if f.endswith('.json')]
    files.sort()

    # Limit the number of frames if the argument was provided
    if num_frames is not None:
        files = files[:num_frames]
            
    print(f"Starting Point-Based Evaluation on {len(files)} frames...\n")
    print("-" * 65)
    
    for file in files:
        pcd_path = os.path.join(pcd_dir, file.replace('.json', '.pcd'))
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        if not os.path.exists(gt_path):
            print(f"[WARNING] Missing GT file for {file}. Skipping...")
            continue
        if not os.path.exists(pcd_path):
            print(f"[WARNING] Missing PCD file for {file}. Skipping...")
            continue
        
        # load point cloud
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        
        # Load JSON labels
        with open(gt_path, 'r') as f:
            gt_boxes = json.load(f)
        with open(pred_path, 'r') as f:
            pred_boxes = json.load(f)
            
        # Temporary dictionaries to hold frame-level class counts
        frame_gt_pts = {}
        frame_pred_pts = {}
        frame_gt_objs = {}
        frame_pred_objs = {}
        
        for box in gt_boxes:
            cls = box.get('obj_type', 'unknown')
            if 'psr' in box:
                pts = get_points_in_box(points, box['psr'])
                frame_gt_pts[cls] = frame_gt_pts.get(cls, 0) + pts
                frame_gt_objs[cls] = frame_gt_objs.get(cls, 0) + 1

        for box in pred_boxes:
            cls = box.get('obj_type', 'unknown')
            if 'psr' in box:
                pts = get_points_in_box(points, box['psr'])
                frame_pred_pts[cls] = frame_pred_pts.get(cls, 0) + pts
                frame_pred_objs[cls] = frame_pred_objs.get(cls, 0) + 1
                
        # find all unique classes present in this frame
        all_classes_in_frame = set(frame_gt_pts.keys()).union(set(frame_pred_pts.keys()))
                
        frame_error = 0
        
        for cls in all_classes_in_frame:
            if cls not in class_metrics:
                class_metrics[cls] = {'gt_pts': 0, 'pred_pts': 0, 'abs_error': 0, 'objs': 0}
                
            c_gt_p = frame_gt_pts.get(cls, 0)
            c_pr_p = frame_pred_pts.get(cls, 0)
            c_gt_o = frame_gt_objs.get(cls, 0)
            c_pr_o = frame_pred_objs.get(cls, 0)
            
            c_err = abs(c_gt_p - c_pr_p)
            c_objs = max(c_gt_o, c_pr_o) # Max accounts for FPs or FNs in object counts
            
            # update class metrics
            class_metrics[cls]['gt_pts'] += c_gt_p
            class_metrics[cls]['pred_pts'] += c_pr_p
            class_metrics[cls]['abs_error'] += c_err
            class_metrics[cls]['objs'] += c_objs
            
            # update global metrics
            global_gt_pts += c_gt_p
            global_pred_pts += c_pr_p
            global_abs_error += c_err
            global_objs += c_objs
            
            frame_total_err += c_err

        print(f"Frame {file} processed | Total Error Diff: {frame_total_err}")

    print("\n" + "="*80)
    print(f"{'CLASS-WISE POINT BENCHMARK TABLE':^80}")
    print("="*80)
    print(f"{'Class Name':<25} | {'Objects':<8} | {'GT Points':<10} | {'Pred Points':<11} | {'Error / Object':<15}")
    print("-" * 80)

    # Print rows sorted alphabetically by class name
    for cls in sorted(class_metrics.keys()):
        metrics = class_metrics[cls]
        objs = metrics['objs']
        err_per_obj = (metrics['abs_error'] / objs) if objs > 0 else 0.0
        
        print(f"{cls:<25} | {objs:<8} | {metrics['gt_pts']:<10} | {metrics['pred_pts']:<11} | {err_per_obj:<15.2f}")

    print("-" * 80)
    
    # Print the Overall row
    global_err_per_obj = (global_abs_error / global_objs) if global_objs > 0 else 0.0
    print(f"{'OVERALL':<25} | {global_objs:<8} | {global_gt_pts:<10} | {global_pred_pts:<11} | {global_err_per_obj:<15.2f}")
    print("="*80 + "\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Bounding Boxes by counting enclosed LiDAR points.")
    parser.add_argument("--pcd_dir", type=str, required=True, help="Path to the directory containing .pcd point cloud files.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to the directory containing Ground Truth .json labels.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to the directory containing Predicted/Exported .json labels.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to benchmark from the start of the sequence. (e.g., 5 or 10)")
    
    args = parser.parse_args()
    
    main(args.pcd_dir, args.gt_dir, args.pred_dir, args.num_frames)