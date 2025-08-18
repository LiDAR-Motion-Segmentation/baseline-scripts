#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.image import message_to_cvimage
from rosbags.typesys import Stores, get_typestore
import cv2

def main():
    parser = argparse.ArgumentParser(description='Extract synchronized camera and LiDAR scans from ROS1 bags')
    parser.add_argument('--bag-file', required=True, help='Input ROS1 bag file')
    parser.add_argument('--camera-topic', required=True, help='Camera image topic name')
    parser.add_argument('--camera-info-topic', required=True, help='Camera info topic name')
    parser.add_argument('--lidar-topic', required=True, help='LiDAR topic name')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--sync-tol', type=float, default=0.1, help='Synchronization tolerance in seconds')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    image_dir = output_dir / 'images'
    lidar_dir = output_dir / 'lidar'
    intrinsics_dir = output_dir / 'intrinsics'
    image_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_dir.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    camera_msgs, caminfo_msgs, lidar_msgs = [], [], []

    with AnyReader([Path(args.bag_file)], default_typestore=typestore) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == args.camera_topic:
                camera_msgs.append((timestamp, rawdata, connection))
            elif connection.topic == args.lidar_topic:
                lidar_msgs.append((timestamp, rawdata, connection))
            elif connection.topic == args.camera_info_topic:
                caminfo_msgs.append((timestamp, rawdata, connection))

        camera_msgs.sort(key=lambda x: x[0])
        caminfo_msgs.sort(key=lambda x: x[0])
        lidar_msgs.sort(key=lambda x: x[0])

        def find_closest(ts, msgs):
            return min(msgs, key=lambda x: abs(x[0] - ts), default=None)

        img_ptr = 0
        sync_pairs = []

        for lidar_ts, lidar_data, lidar_conn in lidar_msgs:
            while img_ptr < len(camera_msgs) - 1 and abs(camera_msgs[img_ptr+1][0] - lidar_ts) < abs(camera_msgs[img_ptr][0] - lidar_ts):
                img_ptr += 1
            if abs(camera_msgs[img_ptr][0] - lidar_ts) <= args.sync_tol * 1e9:
                sync_pairs.append((lidar_ts, lidar_data, lidar_conn, camera_msgs[img_ptr]))

        for idx, (lidar_ts, lidar_data, lidar_conn, (img_ts, img_data, img_conn)) in enumerate(sync_pairs):
            try:
                # --- Process image ---
                msg_img = reader.deserialize(img_data, img_conn.msgtype)

                if 'CompressedImage' in img_conn.msgtype:
                    # Decode JPEG-compressed image with OpenCV
                    np_arr = np.frombuffer(msg_img.data, dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"Failed to decode compressed image at timestamp {img_ts}")
                        continue
                else:
                    # Standard Image message
                    img = message_to_cvimage(msg_img)
                    if img.ndim == 1:
                        if hasattr(msg_img, 'height') and hasattr(msg_img, 'width'):
                            try:
                                channels = 3 if 'rgb' in msg_img.encoding.lower() else 1
                                img = img.reshape((msg_img.height, msg_img.width, channels))
                            except ValueError:
                                print(f"Skipping image reshape failure at timestamp {img_ts}")
                                continue
                        else:
                            print(f"No reshape metadata available at timestamp {img_ts}, skipping image")
                            continue
                    if img.ndim not in [2, 3] or img.size == 0:
                        print(f"Invalid image dimensions {img.shape} at timestamp {img_ts}, skipping image")
                        continue
                    encoding = msg_img.encoding.lower()
                    if encoding == 'mono8' and img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                img_filename = image_dir / f"{lidar_ts}.png"
                cv2.imwrite(str(img_filename), img)
                print(f"Saved image: {img_filename.name}")

                # --- Process camera intrinsics ---
                caminfo = find_closest(img_ts, caminfo_msgs)
                if caminfo:
                    caminfo_msg = reader.deserialize(caminfo[1], caminfo[2].msgtype)
                    K = np.array(caminfo_msg.K).reshape(3, 3)
                    D = np.array(caminfo_msg.D)
                    width = caminfo_msg.width
                    height = caminfo_msg.height
                    intrinsic_filename = intrinsics_dir / f"{lidar_ts}.npz"
                    np.savez(intrinsic_filename, K=K, D=D, width=width, height=height)
                    print(f"Saved intrinsics: {intrinsic_filename.name}")
                else:
                    print(f"No matching camera_info for timestamp {img_ts}")

                # --- Process LiDAR ---
                msg = reader.deserialize(lidar_data, lidar_conn.msgtype)
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

                # Allocate output
                points = np.empty((num_points, 3), dtype=np.float32)

                # Read XYZ fields efficiently
                for i in range(num_points):
                    base = i * point_step
                    points[i, 0] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + x_offset)[0]
                    points[i, 1] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + y_offset)[0]
                    points[i, 2] = np.frombuffer(data, dtype=np.float32, count=1, offset=base + z_offset)[0]

                # Optionally filter invalid points
                valid = ~np.isnan(points).any(axis=1)
                points = points[valid]

                # Save as .npy
                # Save as .pcd using Open3D
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)

                lidar_filename = lidar_dir / f"{lidar_ts}.pcd"
                o3d.io.write_point_cloud(str(lidar_filename), pcd)
                print(f"Saved LiDAR: {lidar_filename.name}")

                print(f"Processed pair {idx+1}/{len(sync_pairs)}")

            except Exception as e:
                print(f"Error processing pair {idx+1}: {e}")

    print(f"Processing complete. Saved {len(sync_pairs)} synchronized pairs.")

if __name__ == "__main__":
    main()