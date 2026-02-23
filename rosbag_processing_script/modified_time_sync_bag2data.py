# made minor changes to orginal code for adaptation with equirectangular images
#!/usr/bin/env python3
import os
import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
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
import open3d as o3d
import numpy as np

# QoS Profiles
qos_profile = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10
)


class BagReader(Node):
    def __init__(self):
        super().__init__("bag_reader")

        self.image1_pub = self.create_publisher(
            Image, "/camera1/camera1/color/image_raw", qos_profile
        )
        self.depth_image1_pub = self.create_publisher(
            Image, "/camera1/camera1/aligned_depth_to_color/image_raw", qos_profile
        )
        self.camera1_info = self.create_publisher(
            CameraInfo, "/camera1/camera1/color/camera_info", qos_profile
        )
        self.image2_pub = self.create_publisher(
            Image, "/camera2/camera2/color/image_raw", qos_profile
        )
        self.depth_image2_pub = self.create_publisher(
            Image, "/camera2/camera2/aligned_depth_to_color/image_raw", qos_profile
        )
        self.camera2_info = self.create_publisher(
            CameraInfo, "/camera2/camera2/color/camera_info", qos_profile
        )
        self.fisheye_pub = self.create_publisher(
            Image, "/dual_fisheye/image", qos_profile
        )
        self.equirectangular_pub = self.create_publisher(
            Image, "/equirectangular/image", qos_profile
        )
        self.point_cloud_pub = self.create_publisher(
            PointCloud2, "/livox/lidar", qos_profile
        )
        self.odom_pub = self.create_publisher(Odometry, "/Odometry", qos_profile)

        self.reader = rosbag2_py.SequentialReader()
        bag_path = os.path.expanduser(
            "/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/2025-10-05_17-06-25/rosbag/rosbag_0.db3"
        )
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions("", "")
        self.reader.open(storage_options, converter_options)

        self.topic = {
            "/camera1/camera1/color/image_raw": (Image, self.image1_pub),
            "/camera1/camera1/aligned_depth_to_color/image_raw": (
                Image,
                self.depth_image1_pub,
            ),
            "/camera1/camera1/color/camera_info": (CameraInfo, self.camera1_info),
            "/camera2/camera2/color/image_raw": (Image, self.image2_pub),
            "/camera2/camera2/aligned_depth_to_color/image_raw": (
                Image,
                self.depth_image2_pub,
            ),
            "/camera2/camera2/color/camera_info": (CameraInfo, self.camera2_info),
            "/dual_fisheye/image": (Image, self.fisheye_pub),
            "/equirectangular/image": (Image, self.equirectangular_pub),
            "/livox/lidar": (PointCloud2, self.point_cloud_pub),
            "/Odometry": (Odometry, self.odom_pub),
        }

        self.create_timer(0.0001, self.ros_play)

        self.played = False

    def ros_play(self):
        if self.played:
            return
        self.played = True
        self.get_logger().info("Started playing bag")
        wall_start_time = time.time()
        bag_start_time = None

        while self.reader.has_next():
            try:
                (topic, data, timestamp) = self.reader.read_next()
                if bag_start_time is None:
                    bag_start_time = timestamp
                intended_time = wall_start_time + (timestamp - bag_start_time) / 1e9
                now = time.time()
                sleep_time = intended_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                if topic in self.topic:
                    message_type, publisher = self.topic[topic]
                    msg = deserialize_message(data, message_type)
                    publisher.publish(msg)

            except Exception as e:
                self.get_logger().error(f"Error reading bag: {e}")
                break


class BagtoData(Node):
    def __init__(self):
        super().__init__("data_converter")

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.save_images1 = "/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/images1"
        # self.save_depth_images1 = '/home/container_user/wheelchair2/src/custom_dataset/data/husky_dataset/depth_images1'
        self.save_images2 = "/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/images2"
        # self.save_depth_images2 = '/home/container_user/wheelchair2/src/custom_dataset/data/husky_dataset/depth_images2'
        self.save_pcd = "/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/pcd"
        self.save_equirectangular = "/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/equirectangular"
        self.save_world2ego = "/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/world2ego"
        self.save_intrinsics = "/media/soumoroy/Extreme SSD/Motion-segementation-rosbags/tech_fest/data/husky_dataset/intrinsics"
        self.frame_count = 0
        self.camera1_intrinsics_saved = False
        self.camera2_intrinsics_saved = False

        os.makedirs(self.save_images1, exist_ok=True)
        # os.makedirs(self.save_depth_images1, exist_ok=True)
        os.makedirs(self.save_images2, exist_ok=True)
        # os.makedirs(self.save_depth_images2, exist_ok=True)
        os.makedirs(self.save_pcd, exist_ok=True)
        os.makedirs(self.save_equirectangular, exist_ok=True)
        os.makedirs(self.save_world2ego, exist_ok=True)
        os.makedirs(self.save_intrinsics, exist_ok=True)

        self.image1_sub = message_filters.Subscriber(
            self, Image, "/camera1/camera1/color/image_raw",
        )
        # self.depth_image1_sub = message_filters.Subscriber(
        #     self,
        #     Image,
        #     "/camera1/camera1/aligned_depth_to_color/image_raw",
        # )
        self.camera1_info = self.create_subscription(
            CameraInfo,
            "/camera1/camera1/color/camera_info",
            self.camera1_info_callback,
            qos_profile,
        )
        self.image2_sub = message_filters.Subscriber(
            self, Image, "/camera2/camera2/color/image_raw",
        )
        # self.depth_image2_sub = message_filters.Subscriber(
        #     self,
        #     Image,
        #     "/camera2/camera2/aligned_depth_to_color/image_raw",
        # )
        self.camera2_info = self.create_subscription(
            CameraInfo,
            "/camera2/camera2/color/camera_info",
            self.camera2_info_callback,
            qos_profile,
        )
        self.fisheye_sub = message_filters.Subscriber(
            self, Image, "/dual_fisheye/image",
        )
        self.equirectangular_sub = message_filters.Subscriber(
            self, Image, "/equirectangular/image",
        )
        self.lidar_sub = message_filters.Subscriber(self, PointCloud2, "/livox/lidar",)
        self.world2ego_sub = message_filters.Subscriber(self, Odometry, "/Odometry",)

        self.time_sync = message_filters.ApproximateTimeSynchronizer(
            [
                self.image1_sub,
                self.image2_sub,
                self.equirectangular_sub,
                self.lidar_sub,
                self.world2ego_sub,
            ],
            queue_size=50,
            slop=0.5,
        )
        self.time_sync.registerCallback(self.synchronized_callback)

    def camera1_info_callback(self, msg):
        if not self.camera1_intrinsics_saved:
            self.camera1_intrinsics = np.array(msg.k)
            filepath = os.path.join(self.save_intrinsics, "1.txt")
            np.savetxt(filepath, self.camera1_intrinsics, fmt="%.6f")
            self.camera1_intrinsics_saved = True

    def camera2_info_callback(self, msg):
        if not self.camera2_intrinsics_saved:
            self.camera2_intrinsics = np.array(msg.k)
            filepath = os.path.join(self.save_intrinsics, "2.txt")
            np.savetxt(filepath, self.camera2_intrinsics, fmt="%.6f")
            self.camera2_intrinsics_saved = True

    def synchronized_callback(
        self, image1_msg, image2_msg, equirectangular_msg, lidar_msg, world2ego_msg
    ):
        cv_image1 = self.bridge.imgmsg_to_cv2(image1_msg, desired_encoding="bgr8")
        cv_image2 = self.bridge.imgmsg_to_cv2(image2_msg, desired_encoding="bgr8")
        cv_equirectangular = self.bridge.imgmsg_to_cv2(
            equirectangular_msg, desired_encoding="bgr8"
        )
        # cv_depth_image1 = self.bridge.imgmsg_to_cv2(depth_image1_msg, desired_encoding="passthrough")
        # cv_depth_image2 = self.bridge.imgmsg_to_cv2(depth_image2_msg, desired_encoding="passthrough")

        cloud_array = ros2_numpy.point_cloud2.pointcloud2_to_array(lidar_msg)
        points = ros2_numpy.point_cloud2.get_xyz_points(cloud_array, remove_nans=True)

        filename_image1 = os.path.join(self.save_images1, f"{self.frame_count:06d}.png")
        # filename_depth_image1 = os.path.join(self.save_depth_images1, f"{self.frame_count:06d}.png")
        filename_image2 = os.path.join(self.save_images2, f"{self.frame_count:06d}.png")
        # filename_depth_image2 = os.path.join(self.save_depth_images2, f"{self.frame_count:06d}.png")
        filename_equirectangular = os.path.join(
            self.save_equirectangular, f"{self.frame_count:06d}.png"
        )

        cv2.imwrite(filename_image1, cv_image1)
        cv2.imwrite(filename_image2, cv_image2)
        # cv2.imwrite(filename_depth_image1, cv_depth_image1)
        # cv2.imwrite(filename_depth_image2, cv_depth_image2)
        cv2.imwrite(filename_equirectangular, cv_equirectangular)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        filename_pcd = os.path.join(self.save_pcd, f"{self.frame_count:06d}.pcd")
        o3d.io.write_point_cloud(filename_pcd, pcd, write_ascii=True)

        translation = world2ego_msg.pose.pose.position
        trans_vec = np.array([translation.x, translation.y, translation.z])

        orientation = world2ego_msg.pose.pose.orientation
        quat_vec = np.array(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        rotation_matrix = ros2_numpy.geometry.transformations.quaternion_matrix(
            quat_vec
        )
        translation_matrix = ros2_numpy.geometry.transformations.translation_matrix(
            trans_vec
        )
        transform_matrix = np.dot(translation_matrix, rotation_matrix)

        world2ego_filename = os.path.join(
            self.save_world2ego, f"{self.frame_count:06d}.txt"
        )
        np.savetxt(world2ego_filename, transform_matrix, fmt="%.18e")

        self.get_logger().info(f"Saved synchronized data for frame {self.frame_count}")

        self.frame_count += 1


def main(args=None):
    RUN_WITH_BAG = True

    rclpy.init(args=args)

    humandetector = BagtoData()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(humandetector)

    bag_reader = None
    if RUN_WITH_BAG:
        print("RUNNING IN BAG PLAYBACK MODE")
        bag_reader = BagReader()
        executor.add_node(bag_reader)
    else:
        print("RUNNING IN LIVE MODE (DETECTOR ONLY)")

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if bag_reader is not None:
            bag_reader.destroy_node()
        humandetector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
