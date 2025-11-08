import os
import argparse
import json
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import ros2_numpy
import cv2
import open3d as o3d
import numpy as np


def save_pcd(msg: PointCloud2, filename: str) -> None:
    try:
        cloud_array = ros2_numpy.point_cloud2.pointcloud2_to_array(msg)
        points = ros2_numpy.point_cloud2.get_xyz_points(cloud_array, remove_nans=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
    except Exception as e:
        print(f"Error saving point cloud {filename}: {e}")


def save_image(msg: Image, filename: str, bridge: CvBridge) -> None:
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(filename, cv_image)
    except Exception as e:
        print(f"Error saving image {filename}: {e}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and synchronize LiDAR and camera data from a ROS 2 bag."
    )

    # Required Arguments
    parser.add_argument(
        "--bag_file",
        type=str,
        required=True,
        help="Path to the ROS 2 bag file (e.g., /path/to/bag/rosbag_0.db3).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the base directory where data will be saved.",
    )
    parser.add_argument(
        "--lidar_topic",
        type=str,
        required=True,
        help="The primary LiDAR topic to synchronize against (e.g., /livox/lidar).",
    )
    parser.add_argument(
        "--image_topics",
        type=str,
        required=True,
        nargs="+",
        help="A list of one or more image topics to synchronize (e.g., /cam1/image /cam2/image).",
    )
    parser.add_argument(
        "--slop",
        type=float,
        default=0.05,
        help="Synchronization time tolerance in seconds. (Default: 0.05s / 50ms)",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=100,
        help="Maximum number of messages to buffer for each topic. (Default: 100)",
    )

    return parser.parse_args()


def create_output_directories(
    output_dir: str, image_topics: List[str]
) -> Tuple[str, Dict[str, str]]:
    print(f"Creating output structure in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    pcd_dir = os.path.join(output_dir, "pointclouds")
    os.makedirs(pcd_dir, exist_ok=True)
    print(f"- Created: {pcd_dir}")

    image_dir_mapping = {}
    for i, topic in enumerate(image_topics, start=1):
        img_dir = os.path.join(output_dir, f"images_{i}")
        os.makedirs(img_dir, exist_ok=True)
        image_dir_mapping[topic] = img_dir
        print(f"  - Mapping '{topic}' -> {img_dir}")

    return pcd_dir, image_dir_mapping


def get_topic_type_map(reader: rosbag2_py.SequentialReader) -> Dict[str, Any]:
    topic_type_map = {}
    for topic_meta in reader.get_all_topics_and_types():
        topic_type_map[topic_meta.name] = get_message(topic_meta.type)
    return topic_type_map


def process_bag(
    args: argparse.Namespace, pcd_dir: str, image_dir_mapping: Dict[str, str]
) -> None:
    print(f"\nOpening bag file: {args.bag_file}")
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=args.bag_file, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    topic_type_map = get_topic_type_map(reader)
    all_topics_of_interest = set([args.lidar_topic] + args.image_topics)
    camera_deques: Dict[str, deque] = {
        topic: deque(maxlen=args.queue_size) for topic in args.image_topics
    }

    bridge = CvBridge()
    records = []
    save_idx = 0
    lidar_msg_count = 0

    print("Starting bag processing and synchronization")

    while reader.has_next():
        topic, raw_data, timestamp_ns = reader.read_next()

        if topic not in all_topics_of_interest:
            continue

        try:
            msg_type = topic_type_map[topic]
            msg = deserialize_message(raw_data, msg_type)
        except Exception as e:
            print(
                f"Warning: Failed to deserialize message on topic {topic}. Skipping. Error: {e}"
            )
            continue

        timestamp_sec = timestamp_ns / 1e9

        if topic in args.image_topics:
            camera_deques[topic].append((timestamp_sec, msg))

        elif topic == args.lidar_topic:
            lidar_msg_count += 1
            best_matches: Dict[str, Tuple] = {}
            all_camera_found = True

            # Check each camera deque for a matching message
            for cam_topic in args.image_topics:
                camera_deque = camera_deques[cam_topic]
                current_best_match = None
                best_delta = float("inf")

                for img_time, img_msg in camera_deque:
                    delta = timestamp_sec - img_time

                    # Check if it's a valid match:
                    # 1. Image must be before or at the same time as LiDAR (delta >= 0)
                    # 2. Image must be within the slop window (delta <= args.slop)
                    # 3. It must be the *closest* match so far (delta < best_delta)

                    if 0 <= delta <= args.slop and delta < best_delta:
                        best_delta = delta
                        current_best_match = (img_time, img_msg, delta)

                if current_best_match:
                    best_matches[cam_topic] = current_best_match
                else:
                    all_camera_found = False
                    break

            if all_camera_found:
                save_idx_str = f"{save_idx:06d}"

                pcd_filename = os.path.join(pcd_dir, f"{save_idx_str}.pcd")
                save_pcd(msg, pcd_filename)

                record_entry = {
                    "frame_id": save_idx,
                    "lidar_timestamp_sec": timestamp_sec,
                    "lidar_pcd_path": os.path.relpath(pcd_filename, args.output_dir),
                    "images": [],
                }

                for cam_topic, (img_time, img_msg, delta) in best_matches.items():
                    img_dir = image_dir_mapping[cam_topic]
                    img_filename = os.path.join(img_dir, f"{save_idx_str}.png")
                    save_image(img_msg, img_filename, bridge)

                    record_entry["images"].append(
                        {
                            "topic": cam_topic,
                            "timestamp_sec": img_time,
                            "path": os.path.relpath(img_filename, args.output_dir),
                            "time_delta_to_lidar": delta,
                        }
                    )

                records.append(record_entry)

                if save_idx % 20 == 0:  # Print status every 20 frames
                    print(f"[OK] Saved synchronized frame {save_idx_str}")

                save_idx += 1

            elif lidar_msg_count % 50 == 0:  # Print a warning if we're failing to sync
                print(
                    f"[Sync Fail] No complete match for LiDAR frame at {timestamp_sec:.4f}s"
                )

    print("\nBag processing complete.")

    # Save the metadata file
    json_path = os.path.join(args.output_dir, "metadata.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=4)

    print(f"\n--- Summary ---")
    print(f"Total LiDAR messages processed: {lidar_msg_count}")
    print(f"Total synchronized frames saved: {save_idx}")
    print(f"Metadata saved to: {json_path}")
    print(f"Data saved in: {args.output_dir}")


def main():
    args = parse_arguments()
    try:
        pcd_dir, image_dir_mapping = create_output_directories(
            args.output_dir, args.image_topics
        )
    except OSError as e:
        print(f"Error: Could not create output directories. {e}")
        return

    process_bag(args, pcd_dir, image_dir_mapping)


if __name__ == "__main__":
    main()
