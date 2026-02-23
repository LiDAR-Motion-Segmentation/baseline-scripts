#!/usr/bin/env python3
# code contributed by @rtarun1

import os
import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from cv_bridge import CvBridge
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
import message_filters
import ros2_numpy
import cv2
import time
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import open3d as o3d
import numpy as np
import json
from collections import deque


def save_pcd(msg, filename):
    cloud = ros2_numpy.point_cloud2.pointcloud2_to_array(msg)
    points = ros2_numpy.point_cloud2.get_xyz_points(cloud, remove_nans=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


def save_image(msg, filename):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    cv2.imwrite(filename, cv_image)


def save_camera_params(msg, intrinsics_path, distortion_path):
    intrinsics = np.array(msg.k).reshape(3, 3)
    np.savetxt(intrinsics_path, intrinsics, fmt="%.6f")
    distortion = np.array(msg.d)
    np.savetxt(distortion_path, distortion, fmt="%.6f")


def main():
    save_path = "data_extraction"
    os.makedirs(os.path.join(save_path, "pcd"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "camera1"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "camera2"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "camera3"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "camera4"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "camera5"), exist_ok=True)

    reader = rosbag2_py.SequentialReader()
    bag_path = os.path.expanduser(
        "/scratch2/soumo_roy/nexus_mall_bags/nexus_lower_ground_2/rosbag/rosbag_0.db3"
    )
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    queue_size = 100
    camera1_queue = deque(maxlen=queue_size)
    camera2_queue = deque(maxlen=queue_size)
    camera3_queue = deque(maxlen=queue_size)
    camera4_queue = deque(maxlen=queue_size)
    camera5_queue = deque(maxlen=queue_size)

    camera1_intrinsics_saved = False
    camera2_intrinsics_saved = False
    camera3_intrinsics_saved = False
    camera4_intrinsics_saved = False
    camera5_intrinsics_saved = False

    records = []
    save_idx = 0
    slop = 0.05
    lidar_count = 0

    CAMERA1_TOPIC = "/camera1/camera1/color/image_raw"
    CAMERA2_TOPIC = "/camera2/camera2/color/image_raw"
    CAMERA3_TOPIC = "/camera3/camera3/color/image_raw"
    CAMERA4_TOPIC = "/camera4/camera4/color/image_raw"
    CAMERA5_TOPIC = "/camera5/camera5/color/image_raw"
    CAMERA1_INFO_TOPIC = "/camera1/camera1/color/camera_info"
    CAMERA2_INFO_TOPIC = "/camera2/camera2/color/camera_info"
    CAMERA3_INFO_TOPIC = "/camera3/camera3/color/camera_info"
    CAMERA4_INFO_TOPIC = "/camera4/camera4/color/camera_info"
    CAMERA5_INFO_TOPIC = "/camera5/camera5/color/camera_info"
    LIDAR_TOPIC = "/livox/lidar"

    while reader.has_next():
        topic, raw, timestamp = reader.read_next()

        if topic == CAMERA1_TOPIC:
            msg = deserialize_message(raw, Image)
            camera1_queue.append((timestamp / 1e9, msg))

        elif topic == CAMERA2_TOPIC:
            msg = deserialize_message(raw, Image)
            camera2_queue.append((timestamp / 1e9, msg))

        elif topic == CAMERA3_TOPIC:
            msg = deserialize_message(raw, Image)
            camera3_queue.append((timestamp / 1e9, msg))

        elif topic == CAMERA4_TOPIC:
            msg = deserialize_message(raw, Image)
            camera4_queue.append((timestamp / 1e9, msg))

        elif topic == CAMERA5_TOPIC:
            msg = deserialize_message(raw, Image)
            camera5_queue.append((timestamp / 1e9, msg))

        elif topic == CAMERA1_INFO_TOPIC and not camera1_intrinsics_saved:
            msg = deserialize_message(raw, CameraInfo)
            intrinsics_path = os.path.join(save_path, "camera1_intrinsics.txt")
            distortion_path = os.path.join(save_path, "camera1_distortion.txt")
            save_camera_params(msg, intrinsics_path, distortion_path)
            camera1_intrinsics_saved = True
            print(f"[INFO] Camera1 intrinsics and distortion saved")

        elif topic == CAMERA2_INFO_TOPIC and not camera2_intrinsics_saved:
            msg = deserialize_message(raw, CameraInfo)
            intrinsics_path = os.path.join(save_path, "camera2_intrinsics.txt")
            distortion_path = os.path.join(save_path, "camera2_distortion.txt")
            save_camera_params(msg, intrinsics_path, distortion_path)
            camera2_intrinsics_saved = True
            print(f"[INFO] Camera2 intrinsics and distortion saved")

        elif topic == CAMERA3_INFO_TOPIC and not camera3_intrinsics_saved:
            msg = deserialize_message(raw, CameraInfo)
            intrinsics_path = os.path.join(save_path, "camera3_intrinsics.txt")
            distortion_path = os.path.join(save_path, "camera3_distortion.txt")
            save_camera_params(msg, intrinsics_path, distortion_path)
            camera3_intrinsics_saved = True
            print(f"[INFO] Camera3 intrinsics and distortion saved")

        elif topic == CAMERA4_INFO_TOPIC and not camera4_intrinsics_saved:
            msg = deserialize_message(raw, CameraInfo)
            intrinsics_path = os.path.join(save_path, "camera4_intrinsics.txt")
            distortion_path = os.path.join(save_path, "camera4_distortion.txt")
            save_camera_params(msg, intrinsics_path, distortion_path)
            camera4_intrinsics_saved = True
            print(f"[INFO] Camera4 intrinsics and distortion saved")

        elif topic == CAMERA5_INFO_TOPIC and not camera5_intrinsics_saved:
            msg = deserialize_message(raw, CameraInfo)
            intrinsics_path = os.path.join(save_path, "camera5_intrinsics.txt")
            distortion_path = os.path.join(save_path, "camera5_distortion.txt")
            save_camera_params(msg, intrinsics_path, distortion_path)
            camera5_intrinsics_saved = True
            print(f"[INFO] Camera5 intrinsics and distortion saved")

        elif topic == LIDAR_TOPIC:
            msg = deserialize_message(raw, PointCloud2)
            lidar_time = timestamp / 1e9

            best_cam1 = None
            best_cam2 = None
            best_cam3 = None
            best_cam4 = None
            best_cam5 = None
            best_delta1 = float("inf")
            best_delta2 = float("inf")
            best_delta3 = float("inf")
            best_delta4 = float("inf")
            best_delta5 = float("inf")

            for cam_time, cam_msg in camera1_queue:
                delta = abs(lidar_time - cam_time)
                if delta <= slop and delta < best_delta1:
                    best_delta1 = delta
                    best_cam1 = (cam_time, cam_msg)

            for cam_time, cam_msg in camera2_queue:
                delta = abs(lidar_time - cam_time)
                if delta <= slop and delta < best_delta2:
                    best_delta2 = delta
                    best_cam2 = (cam_time, cam_msg)

            for cam_time, cam_msg in camera3_queue:
                delta = abs(lidar_time - cam_time)
                if delta <= slop and delta < best_delta3:
                    best_delta3 = delta
                    best_cam3 = (cam_time, cam_msg)

            for cam_time, cam_msg in camera4_queue:
                delta = abs(lidar_time - cam_time)
                if delta <= slop and delta < best_delta4:
                    best_delta4 = delta
                    best_cam4 = (cam_time, cam_msg)

            for cam_time, cam_msg in camera5_queue:
                delta = abs(lidar_time - cam_time)
                if delta <= slop and delta < best_delta5:
                    best_delta5 = delta
                    best_cam5 = (cam_time, cam_msg)

            if all([best_cam1, best_cam2, best_cam3, best_cam4, best_cam5]):

                pcd_path = os.path.join(save_path, "pcd", f"{save_idx:06d}.pcd")
                save_pcd(msg, pcd_path)

                img_path1 = os.path.join(save_path, "camera1", f"{save_idx:06d}.png")
                save_image(best_cam1[1], img_path1)

                img_path2 = os.path.join(save_path, "camera2", f"{save_idx:06d}.png")
                save_image(best_cam2[1], img_path2)

                img_path3 = os.path.join(save_path, "camera3", f"{save_idx:06d}.png")
                save_image(best_cam3[1], img_path3)

                img_path4 = os.path.join(save_path, "camera4", f"{save_idx:06d}.png")
                save_image(best_cam4[1], img_path4)

                img_path5 = os.path.join(save_path, "camera5", f"{save_idx:06d}.png")
                save_image(best_cam5[1], img_path5)

                seconds = msg.header.stamp.sec
                nanoseconds = msg.header.stamp.nanosec

                records.append(
                    {
                        "frame": save_idx,
                        "lidar_sec": seconds,
                        "lidar_nanosec": nanoseconds,
                        "lidar_src_index": lidar_count,
                        "delta_camera1": best_delta1,
                        "delta_camera2": best_delta2,
                        "delta_camera3": best_delta3,
                        "delta_camera4": best_delta4,
                        "delta_camera5": best_delta5,
                        "max_delta": max(
                            best_delta1,
                            best_delta2,
                            best_delta3,
                            best_delta4,
                            best_delta5,
                        ),
                    }
                )

                print(
                    f"[OK] Frame {save_idx:06d} | Δt: cam1={best_delta1:.4f}s, cam2={best_delta2:.4f}s, cam3={best_delta3:.4f}s, cam4={best_delta4:.4f}s, cam5={best_delta5:.4f}s"
                )
                save_idx += 1

            lidar_count += 1

    json_path = os.path.join(save_path, "sync_metadata.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=4)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total frames saved: {save_idx}")
    print(f"Total LiDAR messages processed: {lidar_count}")
    print(f"Metadata saved to: {json_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
