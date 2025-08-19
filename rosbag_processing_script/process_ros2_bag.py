#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
import cv2
import sqlite3
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

def main():
    parser = argparse.ArgumentParser(description='Extract synchronized camera and LiDAR scans from ROS2 bags')
    
    # Required arguments
    parser.add_argument('--bag-file', required=True, help='Input ROS2 bag file (.db3)')
    parser.add_argument('--lidar-topic', required=True, help='LiDAR topic name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    
    # Camera configuration arguments
    parser.add_argument('--camera-topic', required=True, help='Primary camera image topic')
    parser.add_argument('--camera-info-topic', required=True, help='Primary camera info topic')
    parser.add_argument('--camera-topic-2', default=True, help='Secondary camera image topic (optional)')
    parser.add_argument('--camera-info-topic-2', default=True, help='Secondary camera info topic (optional)')
    
    # Processing options
    parser.add_argument('--sync-tol', type=float, default=0.1, help='Synchronization tolerance in seconds')
    parser.add_argument('--dual-camera', action='store_true', help='Enable dual camera processing mode')
    
    args = parser.parse_args()
    
    # Validate dual camera configuration
    if args.dual_camera and (not args.camera_topic_2 or not args.camera_info_topic_2):
        parser.error("--dual-camera requires both --camera-topic-2 and --camera-info-topic-2")

    output_dir = Path(args.output_dir)
    lidar_dir = output_dir / 'lidar'
    
    # Create directories based on camera configuration
    if args.dual_camera:
        image1_dir = output_dir / 'camera1_images'
        intrinsics1_dir = output_dir / 'camera1_intrinsics'
        image2_dir = output_dir / 'camera2_images'
        intrinsics2_dir = output_dir / 'camera2_intrinsics'
        dirs_to_create = [image1_dir, intrinsics1_dir, image2_dir, intrinsics2_dir, lidar_dir]
    else:
        image_dir = output_dir / 'images'
        intrinsics_dir = output_dir / 'intrinsics'
        dirs_to_create = [image_dir, intrinsics_dir, lidar_dir]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)

    # SQLite Database Access
    bag_path = Path(args.bag_file)
    if not (bag_path.is_file() and bag_path.suffix == '.db3'):
        raise FileNotFoundError("Invalid .db3 file path")
    
    conn = sqlite3.connect(str(bag_path))
    cursor = conn.cursor()
    
    # Get topic metadata
    topics_data = cursor.execute("SELECT id, name, type FROM topics").fetchall()
    topic_type_map = {name: type_name for id_val, name, type_name in topics_data}
    topic_id_map = {name: id_val for id_val, name, type_name in topics_data}
    
    print(f"Available topics: {list(topic_type_map.keys())}")
    
    # Collect messages based on configuration
    camera_msgs = []
    caminfo_msgs = []
    camera2_msgs = []
    caminfo2_msgs = []
    lidar_msgs = []
    
    # Primary camera (always required)
    if args.camera_topic in topic_id_map:
        camera_id = topic_id_map[args.camera_topic]
        camera_rows = cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", 
            (camera_id,)).fetchall()
        camera_msgs = [(timestamp, data, topic_type_map[args.camera_topic]) for timestamp, data in camera_rows]
    
    if args.camera_info_topic in topic_id_map:
        caminfo_id = topic_id_map[args.camera_info_topic]
        caminfo_rows = cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", 
            (caminfo_id,)).fetchall()
        caminfo_msgs = [(timestamp, data, topic_type_map[args.camera_info_topic]) for timestamp, data in caminfo_rows]
    
    # Secondary camera (conditional)
    if args.dual_camera and args.camera_topic_2 in topic_id_map:
        camera2_id = topic_id_map[args.camera_topic_2]
        camera2_rows = cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", 
            (camera2_id,)).fetchall()
        camera2_msgs = [(timestamp, data, topic_type_map[args.camera_topic_2]) for timestamp, data in camera2_rows]
    
    if args.dual_camera and args.camera_info_topic_2 in topic_id_map:
        caminfo2_id = topic_id_map[args.camera_info_topic_2]
        caminfo2_rows = cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", 
            (caminfo2_id,)).fetchall()
        caminfo2_msgs = [(timestamp, data, topic_type_map[args.camera_info_topic_2]) for timestamp, data in caminfo2_rows]
    
    # LiDAR messages
    if args.lidar_topic in topic_id_map:
        lidar_id = topic_id_map[args.lidar_topic]
        lidar_rows = cursor.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", 
            (lidar_id,)).fetchall()
        lidar_msgs = [(timestamp, data, topic_type_map[args.lidar_topic]) for timestamp, data in lidar_rows]
    
    conn.close()
    
    print(f"Found {len(camera_msgs)} primary camera messages, {len(lidar_msgs)} LiDAR messages")
    if args.dual_camera:
        print(f"Found {len(camera2_msgs)} secondary camera messages")

    # Synchronize messages
    sync_pairs_primary = synchronize_with_lidar(lidar_msgs, camera_msgs, args.sync_tol)
    
    if args.dual_camera:
        sync_pairs_secondary = synchronize_with_lidar(lidar_msgs, camera2_msgs, args.sync_tol)
        print(f"Found {len(sync_pairs_primary)} primary and {len(sync_pairs_secondary)} secondary sync pairs")
    else:
        print(f"Found {len(sync_pairs_primary)} synchronized pairs")
    
    # Track processed LiDAR timestamps to avoid duplicates
    processed_lidar_timestamps = set()
    
    # Process primary camera pairs
    for idx, (lidar_ts, lidar_data, lidar_type, (img_ts, img_data, img_type)) in enumerate(sync_pairs_primary):
        try:
            # Process primary camera image
            img_msg_class = get_message(img_type)
            msg_img = deserialize_message(img_data, img_msg_class)
            img = process_image(msg_img, img_type)
            
            if img is not None:
                if args.dual_camera:
                    img_filename = image1_dir / f"{lidar_ts}.png"
                else:
                    img_filename = image_dir / f"{lidar_ts}.png"
                cv2.imwrite(str(img_filename), img)
                print(f"Saved primary camera image: {img_filename.name}")

            # Process primary camera intrinsics
            caminfo = find_closest(img_ts, caminfo_msgs)
            if caminfo:
                if args.dual_camera:
                    save_intrinsics(caminfo, intrinsics1_dir, lidar_ts)
                else:
                    save_intrinsics(caminfo, intrinsics_dir, lidar_ts)

            # Process LiDAR (only once per timestamp)
            if lidar_ts not in processed_lidar_timestamps:
                process_and_save_lidar(lidar_data, lidar_type, lidar_dir, lidar_ts)
                processed_lidar_timestamps.add(lidar_ts)

            print(f"Processed primary pair {idx+1}/{len(sync_pairs_primary)}")

        except Exception as e:
            print(f"Error processing primary pair {idx+1}: {e}")

    # Process secondary camera pairs (if dual camera mode)
    if args.dual_camera:
        for idx, (lidar_ts, lidar_data, lidar_type, (img_ts, img_data, img_type)) in enumerate(sync_pairs_secondary):
            try:
                # Process secondary camera image
                img_msg_class = get_message(img_type)
                msg_img = deserialize_message(img_data, img_msg_class)
                img = process_image(msg_img, img_type)
                
                if img is not None:
                    img_filename = image2_dir / f"{lidar_ts}.png"
                    cv2.imwrite(str(img_filename), img)
                    print(f"Saved secondary camera image: {img_filename.name}")

                # Process secondary camera intrinsics
                caminfo2 = find_closest(img_ts, caminfo2_msgs)
                if caminfo2:
                    save_intrinsics(caminfo2, intrinsics2_dir, lidar_ts)

                print(f"Processed secondary pair {idx+1}/{len(sync_pairs_secondary)}")

            except Exception as e:
                print(f"Error processing secondary pair {idx+1}: {e}")

    if args.dual_camera:
        print(f"Processing complete. Primary: {len(sync_pairs_primary)}, Secondary: {len(sync_pairs_secondary)} pairs")
    else:
        print(f"Processing complete. Saved {len(sync_pairs_primary)} synchronized pairs")

def synchronize_with_lidar(lidar_msgs, camera_msgs, sync_tol):
    """Synchronize camera messages with LiDAR messages"""
    img_ptr = 0
    sync_pairs = []
    
    for lidar_ts, lidar_data, lidar_type in lidar_msgs:
        while img_ptr < len(camera_msgs) - 1 and \
              abs(camera_msgs[img_ptr+1][0] - lidar_ts) < abs(camera_msgs[img_ptr][0] - lidar_ts):
            img_ptr += 1
        
        if abs(camera_msgs[img_ptr][0] - lidar_ts) <= sync_tol * 1e9:
            sync_pairs.append((lidar_ts, lidar_data, lidar_type, camera_msgs[img_ptr]))
    
    return sync_pairs

def process_image(msg_img, img_type):
    """Process image message and return OpenCV image"""
    if 'CompressedImage' in img_type:
        np_arr = np.frombuffer(msg_img.data, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    else:
        img = ros_image_to_opencv(msg_img)
        if img.ndim == 1:
            if hasattr(msg_img, 'height') and hasattr(msg_img, 'width'):
                try:
                    channels = 3 if 'rgb' in msg_img.encoding.lower() else 1
                    img = img.reshape((msg_img.height, msg_img.width, channels))
                except ValueError:
                    return None
            else:
                return None
        
        if img.ndim not in [2, 3] or img.size == 0:
            return None
            
        encoding = msg_img.encoding.lower()
        if encoding == 'mono8' and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img

def save_intrinsics(caminfo, intrinsics_dir, timestamp):
    """Save camera intrinsics"""
    caminfo_msg_class = get_message(caminfo[2])
    caminfo_msg = deserialize_message(caminfo[1], caminfo_msg_class)
    K = np.array(caminfo_msg.k).reshape(3, 3)
    D = np.array(caminfo_msg.d)
    width = caminfo_msg.width
    height = caminfo_msg.height
    intrinsic_filename = intrinsics_dir / f"{timestamp}.npz"
    np.savez(intrinsic_filename, K=K, D=D, width=width, height=height)

def process_and_save_lidar(lidar_data, lidar_type, lidar_dir, timestamp):
    """Process and save LiDAR data"""
    lidar_msg_class = get_message(lidar_type)
    msg = deserialize_message(lidar_data, lidar_msg_class)
    num_points = msg.width * msg.height
    point_step = msg.point_step
    data = msg.data

    # Find offsets
    x_offset = y_offset = z_offset = None
    for field in msg.fields:
        if field.name == 'x':
            x_offset = field.offset
        elif field.name == 'y':
            y_offset = field.offset
        elif field.name == 'z':
            z_offset = field.offset

    if None in (x_offset, y_offset, z_offset):
        raise ValueError("Missing x/y/z fields in PointCloud2 message")

    points = np.empty((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        base = i * point_step
        points[i, 0] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + x_offset)[0]
        points[i, 1] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + y_offset)[0]
        points[i, 2] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + z_offset)[0]

    valid = ~np.isnan(points).any(axis=1)
    points = points[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    lidar_filename = lidar_dir / f"{timestamp}.pcd"
    o3d.io.write_point_cloud(str(lidar_filename), pcd)

def find_closest(ts, msgs):
    return min(msgs, key=lambda x: abs(x[0] - ts), default=None)

def ros_image_to_opencv(ros_image):
    """Convert ROS Image message to OpenCV format"""
    if ros_image.encoding == 'bgr8':
        img = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width, 3)
    elif ros_image.encoding == 'rgb8':
        img = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width, 3)
    elif ros_image.encoding == 'mono8':
        img = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(ros_image.height, ros_image.width)
    else:
        raise ValueError(f"Unsupported image encoding: {ros_image.encoding}")
    return img

if __name__ == "__main__":
    main()