# @rtaun1 code
import cv2
import numpy as np
import open3d as o3d
import ros2_numpy
import time
import os


def process_frame(timestep):
    pc_data = o3d.io.read_point_cloud(
        f"/home/container_user/wheelchair2/src/image_video_object_tracking/sam_lidar_output/{timestep:06d}.pcd"
    )
    xyz_points = np.asarray(pc_data.points)
    timestep_p = timestep + 3
    image = cv2.imread(
        f"/home/container_user/wheelchair2/src/image_video_object_tracking/equirectangular/{timestep_p:06d}.png"
    )

    filepath = f"/home/container_user/wheelchair2/src/image_video_object_tracking/my_results_v2/labels_txt/{timestep_p:06d}.txt"
    all_polygons = []
    with open(filepath, "r") as f:
        for line in f:
            poly_data = np.array([float(x) for x in line.split()])
            all_polygons.append(poly_data)

    overlay = image.copy()
    h, w, _ = image.shape
    segmentation_mask = np.zeros((h, w), dtype=np.uint8)

    for poly_data in all_polygons:
        normalized_points = poly_data[1:].reshape(-1, 2)
        pixel_points = (normalized_points * np.array([w, h])).astype(np.int32)
        cv2.fillPoly(overlay, [pixel_points], color=(0, 0, 255))
        cv2.fillPoly(segmentation_mask, [pixel_points], color=1)
    alpha = 0.5
    final_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    output_filename = f"./sam_image/{timestep:06d}.png"
    cv2.imwrite(output_filename, final_image)

    T_lidar_camera_arr = [
        0.0020650576972357916,
        -0.1723896747289117,
        0.06931782278326322,
        -0.5000309238514344,
        0.5035259747293952,
        -0.5018388910867259,
        0.49455878857617996,
    ]

    translation_m = ros2_numpy.geometry.transformations.translation_matrix(
        T_lidar_camera_arr[0:3]
    )
    rotation_m = ros2_numpy.geometry.transformations.quaternion_matrix(
        T_lidar_camera_arr[3:7]
    )
    T_lidar_camera = np.dot(translation_m, rotation_m)
    transform_matrix = np.linalg.inv(T_lidar_camera)

    points_homogeneous = np.hstack([xyz_points, np.ones((xyz_points.shape[0], 1))])
    points_camera_frame = (transform_matrix @ points_homogeneous.T).T[:, :3]

    image_height, image_width, _ = image.shape

    x = points_camera_frame[:, 0]
    y = points_camera_frame[:, 1]
    z = points_camera_frame[:, 2]

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    valid_indices = r > 0
    p_x, p_y, p_z = (
        x[valid_indices] / r[valid_indices],
        y[valid_indices] / r[valid_indices],
        z[valid_indices] / r[valid_indices],
    )
    phi = np.arcsin(p_y)
    theta = np.arctan2(p_x, p_z)

    u_coords = (theta * image_width / (2 * np.pi)) + (image_width / 2)
    v_coords = (phi * image_height / np.pi) + (image_height / 2)

    pixel_coords = np.vstack((u_coords, v_coords)).T

    on_image_mask = (
        (pixel_coords[:, 0] >= 0)
        & (pixel_coords[:, 0] < image_width)
        & (pixel_coords[:, 1] >= 0)
        & (pixel_coords[:, 1] < image_height)
    )

    valid_pixels = pixel_coords[on_image_mask].astype(int)

    mask_values = segmentation_mask[valid_pixels[:, 1], valid_pixels[:, 0]]
    person_mask_on_image = mask_values == 1
    # person_pixels = valid_pixels[person_mask_on_image]

    num_total_points = len(xyz_points)
    colors = np.full((num_total_points, 3), [0.0, 1.0, 0.0])

    original_indices = np.arange(num_total_points)
    indices_on_image = original_indices[valid_indices][on_image_mask]

    person_indices = indices_on_image[person_mask_on_image]

    colors[person_indices] = [1.0, 0.0, 0.0]

    person_points = xyz_points[person_indices]
    save_path = f"./sam_lidar_segmented/{timestep:06d}.npy"
    np.save(save_path, person_points)

    colored_pcd = o3d.geometry.PointCloud()
    colored_pcd.points = o3d.utility.Vector3dVector(xyz_points)
    colored_pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(
        f"./sam_lidar/{timestep:06d}.pcd", colored_pcd,
    )

    return colored_pcd


def update_visualization(vis, point_cloud, new_pcd):
    if point_cloud is None:
        point_cloud = new_pcd
        vis.add_geometry(point_cloud)
    else:
        point_cloud.points = new_pcd.points
        point_cloud.colors = new_pcd.colors
        vis.update_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    return point_cloud


if __name__ == "__main__":
    os.makedirs(
        "./sam_lidar/", exist_ok=True,
    )
    os.makedirs(
        "./sam_image/", exist_ok=True,
    )
    os.makedirs(
        "./sam_lidar_segmented/", exist_ok=True,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Segmented PCD", width=1280, height=720)
    point_cloud = None

    for i in range(1189):
        new_pcd = process_frame(i)
        point_cloud = update_visualization(vis, point_cloud, new_pcd)

    vis.run()
    vis.destroy_window()
