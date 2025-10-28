#!/usr/bin/env python3
# contributed by @GauravKumar9920

import argparse, os, re, shutil, json
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def pick_first_npz(npz_dir):
    files = sorted(glob(os.path.join(npz_dir, "*.npz")))
    return files[0] if files else None


def load_intrinsics(npz_path, sample_img):
    # Try common key names
    K_keys = ["K", "camera_matrix", "intrinsic_matrix"]
    D_keys = ["D", "dist_coeffs", "distortion", "distortion_coefficients"]

    data = np.load(npz_path)
    K = None
    for k in K_keys:
        if k in data:
            K = data[k]
            break
    if K is None:
        raise ValueError(f"No camera matrix key found in {npz_path}. Tried {K_keys}")

    D = None
    for d in D_keys:
        if d in data:
            D = data[d]
            break
    # Normalize shapes
    K = np.array(K).reshape(3, 3)
    if D is not None:
        D = np.array(D).flatten().tolist()
    else:
        D = []

    # Get image size
    with Image.open(sample_img) as im:
        width, height = im.size

    intr = {
        "width": int(width),
        "height": int(height),
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "skew": float(K[0, 1]),  # often 0
        "distortion": D,  # e.g., [k1,k2,p1,p2,k3,...] if present
    }
    return intr


def copy_and_reindex(src_files, dst_dir, ext_preserve=True, force_ext=None):
    """Copy files to dst_dir as zero-padded sequential names. Optionally force a file extension (e.g., '.pcd'). Returns list of new basenames."""
    ensure_dir(dst_dir)
    # fall back to mtime if filenames aren’t sortable
    try:
        src_files = sorted(src_files, key=natural_key)
    except Exception:
        src_files = sorted(src_files, key=lambda p: os.path.getmtime(p))

    new_names = []
    pad = max(6, len(str(len(src_files))))
    for i, src in enumerate(src_files):
        ext = Path(src).suffix if ext_preserve else ""
        if force_ext:
            ext = force_ext
        dst_name = f"{i:0{pad}d}{ext}"
        shutil.copy2(src, os.path.join(dst_dir, dst_name))
        new_names.append(dst_name)
    return new_names


def write_list(fp, names):
    with open(fp, "w") as f:
        for n in names:
            f.write(n + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        required=True,
        help="Source sequence folder (e.g., aug5-group-... path containing camera* and lidar)",
    )
    ap.add_argument(
        "--dst",
        required=True,
        help="Destination DATA ROOT folder (e.g., /Users/gaurav/SUSTechPOINTS/local_data)",
    )
    ap.add_argument(
        "--scene",
        default=None,
        help="Scene folder name to create under --dst (default: derived from --src basename)",
    )
    ap.add_argument(
        "--pcd_only", action="store_true", help="Index only .pcd files (ignore .bin)"
    )
    ap.add_argument(
        "--force_pc_ext",
        choices=["keep", "pcd", "bin"],
        default="keep",
        help="Rename point-cloud extension after indexing: keep original, or force to .pcd/.bin",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst_root = Path(args.dst)

    # Derive a clean scene name if not provided
    scene_name = args.scene
    if not scene_name:
        # Lowercase, replace non-alnum with underscore
        scene_name = re.sub(r"[^a-zA-Z0-9_]+", "_", src.name.lower()).strip("_")
        if not scene_name:
            scene_name = "scene"

    dst_scene = dst_root / scene_name

    # Source folders expected in original input
    img1_dir = src / "camera1_images"
    img2_dir = src / "camera2_images"
    int1_dir = src / "camera1_intrinsics"
    int2_dir = src / "camera2_intrinsics"
    lidar_dir = src / "lidar"

    # Canonical output layout expected by SUSTechPOINTS/scene_reader.py
    out_cam1 = dst_scene / "camera" / "CAMERA1"
    out_cam2 = dst_scene / "camera" / "CAMERA2"
    out_lidar = dst_scene / "lidar"
    out_cal_cam = dst_scene / "calib" / "camera"
    out_label = dst_scene / "label"

    ensure_dir(out_cam1)
    ensure_dir(out_cam2)
    ensure_dir(out_lidar)
    ensure_dir(out_cal_cam)
    ensure_dir(out_label)

    # Gather files
    cam1 = sorted(glob(str(img1_dir / "*.png")), key=natural_key)
    cam2 = sorted(glob(str(img2_dir / "*.png")), key=natural_key)
    pc_patterns = ["*.pcd"] if args.pcd_only else ["*.pcd", "*.bin"]
    pcds = []
    for pat in pc_patterns:
        pcds.extend(glob(str(lidar_dir / pat)))
    pcds = sorted(pcds, key=natural_key)

    if not cam1:
        print("WARNING: no CAMERA1 PNGs found.")
    if not cam2:
        print("WARNING: no CAMERA2 PNGs found.")
    if not pcds:
        print("WARNING: no LiDAR .pcd/.bin found.")

    # Reindex & copy
    cam1_new = copy_and_reindex(cam1, out_cam1, ext_preserve=True) if cam1 else []
    cam2_new = copy_and_reindex(cam2, out_cam2, ext_preserve=True) if cam2 else []

    force_ext = None
    if args.force_pc_ext == "pcd":
        force_ext = ".pcd"
    elif args.force_pc_ext == "bin":
        force_ext = ".bin"
    pc_new = (
        copy_and_reindex(
            pcds, out_lidar, ext_preserve=(force_ext is None), force_ext=force_ext
        )
        if pcds
        else []
    )

    # Lists (optional but useful for quick checks)
    if cam1_new:
        write_list(
            dst_scene / "CAMERA1_filenames.txt",
            [f"camera/CAMERA1/{n}" for n in cam1_new],
        )
    if cam2_new:
        write_list(
            dst_scene / "CAMERA2_filenames.txt",
            [f"camera/CAMERA2/{n}" for n in cam2_new],
        )
    if pc_new:
        # Although historically named point_cloud_filenames, we now list lidar paths
        write_list(
            dst_scene / "point_cloud_filenames.txt", [f"lidar/{n}" for n in pc_new]
        )
        # Matching annotation list for convenience
        write_list(
            dst_scene / "annotation_filenames.txt",
            [f"label/{Path(n).stem}.json" for n in pc_new],
        )

    # Intrinsics to JSON
    c1_npz = pick_first_npz(int1_dir) if int1_dir.exists() else None
    c2_npz = pick_first_npz(int2_dir) if int2_dir.exists() else None

    if c1_npz and cam1_new:
        intr1 = load_intrinsics(c1_npz, os.path.join(out_cam1, cam1_new[0]))
        with open(out_cal_cam / "CAMERA1.json", "w") as f:
            json.dump(intr1, f, indent=2)
    else:
        print("NOTE: Skipping CAMERA1 intrinsics (no .npz or no images).")

    if c2_npz and cam2_new:
        intr2 = load_intrinsics(c2_npz, os.path.join(out_cam2, cam2_new[0]))
        with open(out_cal_cam / "CAMERA2.json", "w") as f:
            json.dump(intr2, f, indent=2)
    else:
        print("NOTE: Skipping CAMERA2 intrinsics (no .npz or no images).")

    # Ensure a blank label exists per lidar frame so the UI can save immediately
    for n in pc_new:
        base = Path(n).stem
        lab = out_label / f"{base}.json"
        if not lab.exists():
            with open(lab, "w") as f:
                f.write("[]")

    # Basic alignment report
    print("\n=== Prep summary ===")
    print(f"Destination data root : {dst_root}")
    print(f"Scene name            : {scene_name}")
    print(f"Scene path            : {dst_scene}")
    print(f"CAMERA1 frames        : {len(cam1_new)}")
    print(f"CAMERA2 frames        : {len(cam2_new)}")
    print(f"LiDAR frames          : {len(pc_new)}")
    min_len = min(
        [x for x in [len(cam1_new), len(cam2_new), len(pc_new)] if x > 0], default=0
    )
    if min_len > 0 and (len(cam1_new) != len(cam2_new) or len(cam1_new) != len(pc_new)):
        print(
            f"NOTE: Counts mismatch; you may annotate using the common prefix of ~{min_len} frames."
        )
        print("      (This script preserves ALL frames; you can trim later if needed.)")
    if pcds:
        print(f"Point-cloud extension policy: {args.force_pc_ext}")


if __name__ == "__main__":
    main()
