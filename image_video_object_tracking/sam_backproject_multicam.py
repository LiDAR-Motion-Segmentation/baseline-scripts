import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def load_intrinsics(file_path):
    return np.loadtxt(file_path).reshape(3, 3)


def load_extrinsics(file_path):
    # Load [x,y,z,qx,qy,qz,qw] -> 4x4 matrix.
    arr = np.loadtxt(file_path)
    assert arr.shape == (7,)
    x, y, z, qx, qy, qz, qw = arr
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [x, y, z]
    return T


def parse_label_dirs(label_args):
    label_map = {}
    for cam_label in label_args:
        if ":" not in cam_label:
            raise ValueError(
                f"Label argument '{cam_label}' must be camera_name:/full/path"
            )
        cam_name, dir_path = cam_label.split(":", 1)
        label_map[cam_name] = Path(dir_path)
    return label_map


# def parse_camera_args(camera_args):
#     """
#     camera_args is a list of triples:
#     [cam_name, intr_dir, extr_dir, cam_name2, intr_dir2, extr_dir2, ...]
#     """
#     if len(camera_args) % 4 != 0:
#         raise ValueError("Each --camera argument requires exactly 4 values: name, img_dir, intr_path, extr_path")
#     cameras = []
#     for i in range(0, len(camera_args), 4):
#         cam_name = camera_args[i]
#         img_dir = Path(camera_args[i+1])
#         intr_path = Path(camera_args[i+2])
#         extr_path = Path(camera_args[i+3])
#         if not img_dir.is_dir():
#             raise ValueError(f"Image directory {img_dir} does not exist")
#         if not intr_path.is_file():
#             raise ValueError(f"Intrinsics file {intr_path} does not exist")
#         if not extr_path.is_file():
#             raise ValueError(f"Extrinsics file {extr_path} does not exist")
#         cameras.append({
#             'name': cam_name,
#             'image_dir': img_dir,
#             'intrinsics': load_intrinsics(intr_path),
#             'extrinsics': load_extrinsics(extr_path)
#         })
#     return cameras


def enumerate_cameras(camera_args, label_map):
    if len(camera_args) % 4 != 0:
        raise ValueError(
            "Each --camera argument requires exactly 4 values: name, img_dir, intr_path, extr_path"
        )
    cameras = []
    for i in range(0, len(camera_args), 4):
        cam_name = camera_args[i]
        img_dir = Path(camera_args[i + 1])
        intr_path = Path(camera_args[i + 2])
        extr_path = Path(camera_args[i + 3])
        if not img_dir.is_dir():
            raise ValueError(f"Image directory {img_dir} does not exist")
        if not intr_path.is_file():
            raise ValueError(f"Intrinsics file {intr_path} does not exist")
        if not extr_path.is_file():
            raise ValueError(f"Extrinsics file {extr_path} does not exist")
        intrinsics = load_intrinsics(intr_path)
        extrinsics = load_extrinsics(extr_path)
        label_dir = label_map.get(cam_name, None)
        if label_dir and not label_dir.exists():
            print(f"Warning: Label dir {label_dir} for {cam_name} not found.")
            label_dir = None
        cameras.append(
            {
                "name": cam_name,
                "image_dir": img_dir,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "label_dir": label_dir,
            }
        )
    return cameras


def build_mask_from_label(label_file, image_shape):
    # Given a label (polygon txt) and image shape, build binary mask
    h, w = image_shape[:2]
    segementation_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        with open(label_file, "r") as f:
            for line in f:
                poly_data = np.array([float(x) for x in line.split()])
                if len(poly_data) >= 7:
                    norm_pts = poly_data[1:].reshape(-1, 2)
                    pixel_pts = (norm_pts * np.array([w, h])).astype(np.int32)
                    if pixel_pts.shape[0] >= 3:
                        cv2.fillPoly(segementation_mask, [pixel_pts], color=1)
    except FileNotFoundError:
        print(f"Label not found: {label_file}")
    return segementation_mask


def project_points_to_img(points, T_lidar_cam, K):
    pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
    T_inv = np.linalg.inv(T_lidar_cam)
    pts_cam = (T_inv @ pts_h.T).T[:, :3]
    valid_cam = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid_cam]
    proj_pts = (K @ pts_cam.T).T
    pixel_uv = proj_pts[:, :2] / proj_pts[:, 2:3]
    return pixel_uv, valid_cam


def back_project_mask_to_lidar(points, mask, pixel_uv, valid_cam):
    h, w = mask.shape[:2]
    pixel_uv_int = np.round(pixel_uv).astype(int)
    in_img = (
        (pixel_uv_int[:, 0] >= 0)
        & (pixel_uv_int[:, 0] < w)
        & (pixel_uv_int[:, 1] >= 0)
        & (pixel_uv_int[:, 1] < h)
    )
    indices = np.where(valid_cam)[0][in_img]  # original LiDAR idx for points in FOV
    pixel_uv_on_img = pixel_uv_int[in_img]

    # masked points
    if len(pixel_uv_on_img) == 0:
        return np.array([], dtype=int)
    mask_vals = mask[pixel_uv_on_img[:, 1], pixel_uv_on_img[:, 0]]
    mask_indices = indices[mask_vals == 1]
    return mask_indices


def process_frame_multicam(pcd_path, cameras, output_dirs):
    timestep = pcd_path.stem
    pc = o3d.io.read_point_cloud(str(pcd_path))
    xyz = np.asarray(pc.points)
    all_mask_indices = set()

    for cam in cameras:
        img_files = sorted(cam["image_dir"].glob("*.png")) + sorted(
            cam["image_dir"].glob("*.jpg")
        )
        img_file = next((img for img in img_files if img.stem == timestep), None)
        if not img_file:
            continue
        img = cv2.imread(str(img_file))
        # if img is None:
        #     continue

        label_file = None
        if cam.get("label_dir"):
            label_candidates = list(cam["label_dir"].glob(f"{timestep}*.txt"))
            if label_candidates:
                label_file = label_candidates[0]
        if not label_file:
            continue

        mask = build_mask_from_label(label_file, img.shape)

        overlay = img.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)
        alpha = 0.5
        blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        seg_img_dir = output_dirs["segmented_images"] / cam["name"]
        seg_img_dir.mkdir(parents=True, exist_ok=True)
        seg_img_path = seg_img_dir / f"{timestep}.png"
        cv2.imwrite(str(seg_img_path), blended)
        print(f"[{cam['name']}] Saved segmented image: {seg_img_path}")

        pixel_uv, valid_cam = project_points_to_img(
            xyz, cam["extrinsics"], cam["intrinsics"]
        )
        li_point_idx = back_project_mask_to_lidar(xyz, mask, pixel_uv, valid_cam)
        all_mask_indices.update(li_point_idx.tolist())

    colors = np.full((len(xyz), 3), [0.0, 1.0, 0.0])  # green
    colors[list(all_mask_indices)] = [1.0, 0.0, 0.0]  # red for segmented mask
    colored_pc = o3d.geometry.PointCloud()
    colored_pc.points = o3d.utility.Vector3dVector(xyz)
    colored_pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(output_dirs["pcd"] / f"{timestep}.pcd"), colored_pc)
    print(f"[ok] Saved pointcloud: {output_dirs['pcd']}/{timestep}.pcd")


def main():
    parser = argparse.ArgumentParser(
        description="SAM mask LiDAR backprojection with custom mask label folders per camera."
    )
    parser.add_argument(
        "--camera",
        nargs=4,
        action="append",
        required=True,
        help="Specify one camera config per argument: <cam_name> <image_dir> <intrinsics_file> <extrinsics_file>",
    )
    parser.add_argument("--pcd_dir", type=str, default="pcd")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Label directories as camera_name:/full/path/to/labels (repeat for all)",
    )
    args = parser.parse_args()

    camera_args = [item for sublist in args.camera for item in sublist]
    labels_map = parse_label_dirs(args.labels)
    cameras = enumerate_cameras(camera_args, labels_map)
    output_dirs = {
        "pcd": Path(args.output_dir) / "sam_lidar_segmented",
        "segmented_images": Path(args.output_dir) / "segmented_images",
    }
    output_dirs["pcd"].mkdir(parents=True, exist_ok=True)
    output_dirs["segmented_images"].mkdir(parents=True, exist_ok=True)
    pcd_files = sorted((Path(args.pcd_dir) / args.pcd_dir).glob("*.pcd"))

    for pcd_file in pcd_files:
        try:
            process_frame_multicam(pcd_file, cameras, output_dirs)
        except Exception as e:
            print(f"[err] {pcd_file.name}: {e}")


if __name__ == "__main__":
    main()
