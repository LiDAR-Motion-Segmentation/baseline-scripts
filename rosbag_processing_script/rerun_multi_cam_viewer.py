import os
import argparse
from pathlib import Path
import rerun as rr
import cv2


def enumerate_frames(camera_folders):
    cam_img_map = {}
    frame_ids_all = set()
    for cam_name, cam_dir in camera_folders.items():
        cam_dir = Path(cam_dir)
        images = sorted(cam_dir.glob("*.png")) + sorted(cam_dir.glob("*.jpg"))
        frame_map = {img.stem: img for img in images}
        cam_img_map[cam_name] = frame_map
        frame_ids_all.update(frame_map.keys())
    frames = sorted(frame_ids_all)
    return frames, cam_img_map


def load_images_for_frame(cam_img_map, frame_id):
    images = {}
    for cam_name, frame_map in cam_img_map.items():
        img_path = frame_map.get(frame_id, None)
        if img_path and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                images[cam_name] = img
    return images


def log_rerun_images(images, frame_idx):
    rr.set_time_sequence("frame", frame_idx)
    for cam_name, img in images.items():
        rr.log(f"cam/{cam_name}/image", rr.Image(img[..., ::-1]))  # BGR2RGB


def main():
    parser = argparse.ArgumentParser(
        description="Multi-camera folder Rerun visualization"
    )
    # You can add as many --cam_dirs CAM_NAME PATH as you want
    parser.add_argument(
        "--cam_dirs",
        nargs=2,
        action="append",
        metavar=("CAM_NAME", "PATH"),
        help="Camera name and folder, can be repeated (e.g. --cam_dirs cam0 path0 --cam_dirs cam1 path1 ...)",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0, help="Playback FPS (optional)"
    )
    args = parser.parse_args()

    if not args.cam_dirs:
        print("Error: At least one --cam_dirs CAM_NAME PATH argument is required")
        return
    camera_folders = {name: path for name, path in args.cam_dirs}

    for cam, d in camera_folders.items():
        if not Path(d).is_dir():
            raise ValueError(f"Provided camera directory {d} for {cam} does not exist")

    frames, cam_img_map = enumerate_frames(camera_folders)
    print(f"Found {len(frames)} unique frames, {len(camera_folders)} cameras.")

    rr.init("multi_cam_viewer", spawn=True)
    for idx, frame_id in enumerate(frames):
        images = load_images_for_frame(cam_img_map, frame_id)
        if images:
            log_rerun_images(images, idx)
            print(f"Frame {frame_id}: {[f'{k}' for k in images.keys()]}")
        else:
            print(f"Frame {frame_id}: No images found for any camera.")

    print("Visualization loop complete.")


if __name__ == "__main__":
    main()
