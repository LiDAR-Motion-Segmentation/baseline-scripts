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
            
        frame_gt_pts = 0
        for box in pred_boxes:
            if 'psr' in box:
                frame_gt_pts += get_points_in_box(points, box['psr'])
                
        frame_pred_pts = 0
        for box in pred_boxes:
            if 'psr' in box:
                frame_pred_pts += get_points_in_box(points, box['psr'])
                
        frame_error = abs(frame_gt_pts - frame_pred_pts)
        
        total_gt_points += frame_gt_pts
        total_pred_points += frame_pred_pts
        total_absolute_error += frame_error
        total_objects_evaluated += max(len(gt_boxes), len(pred_boxes))

        print(f"Frame {file} | GT Points: {frame_gt_pts:5} | Pred Points: {frame_pred_pts:5} | Error Diff: {frame_error}")
        
    # Calculate final metrics for the table
    mean_points_error = (total_absolute_error / total_objects_evaluated) if total_objects_evaluated > 0 else 0
    
    # Print final results
    print("\n" + "="*50)
    print("             POINT ERROR BENCHMARK             ")
    print("="*50)
    print(f"Total Frames Evaluated     : {len(files)}")
    print(f"Total Objects Evaluated    : {total_objects_evaluated}")
    print(f"Total GT Points Captured   : {total_gt_points}")
    print(f"Total Pred Points Captured : {total_pred_points}")
    print(f"Total Absolute Point Error : {total_absolute_error}")
    print("-" * 50)
    print(f"Errors (Points/Object)     : {mean_points_error:.2f}")
    print("="*50)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D Bounding Boxes by counting enclosed LiDAR points.")
    parser.add_argument("--pcd_dir", type=str, required=True, help="Path to the directory containing .pcd point cloud files.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to the directory containing Ground Truth .json labels.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Path to the directory containing Predicted/Exported .json labels.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to benchmark from the start of the sequence. (e.g., 5 or 10)")
    
    args = parser.parse_args()
    
    main(args.pcd_dir, args.gt_dir, args.pred_dir, args.num_frames)