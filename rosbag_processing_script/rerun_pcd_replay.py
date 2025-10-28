#!/usr/bin/env python3
"""
Stream a sequence of labeled .pcd point clouds to rerun.io in real time.
"""

import argparse
import glob
import os
import time
import numpy as np
import open3d as o3d
import rerun as rr


def parse_timestamp(filename):
    # Filename format: 1754391085631999872.pcd
    base = os.path.basename(filename)
    stem, _ = os.path.splitext(base)
    try:
        return int(stem)
    except ValueError:
        return None


def main(pcd_folder, start_index=0):
    # Collect and sort PCD files
    files = glob.glob(os.path.join(pcd_folder, "*.pcd"))
    frames = [(parse_timestamp(f), f) for f in files]
    frames = [(ts, f) for ts, f in frames if ts is not None]
    frames.sort(key=lambda x: x[0])

    if not frames:
        print(f"No .pcd files found in {pcd_folder}")
        return

    print(f"Found {len(frames)} PCD files")
    if start_index >= len(frames):
        print(
            f"Start index {start_index} is beyond available frames (0-{len(frames)-1})"
        )
        return

    # Initialize Rerun
    rr.init("pcd_replay", spawn=True)

    last_ts = None
    for idx, (ts, path) in enumerate(frames[start_index:], start_index):
        print(f"Streaming frame {idx}/{len(frames)-1}, timestamp {ts}")

        try:
            # Load point cloud
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points)

            if len(pts) == 0:
                print(f"  Warning: Empty point cloud in {os.path.basename(path)}")
                continue

            colors = None
            if pcd.has_colors():
                colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

            # Set timeline
            rr.set_time_nanos("timestamp", ts)

            # Send to rerun
            rr.log("scene/labeled_cloud", rr.Points3D(positions=pts, colors=colors))

            print(f"  Logged {len(pts)} points")

        except Exception as e:
            print(f"  Error processing {path}: {e}")
            continue

        # Play back at real intervals (but cap the delay)
        if last_ts is not None:
            dt = (ts - last_ts) / 1e9  # ns → seconds
            sleep_time = max(0.0, min(dt, 0.5))  # Cap at 0.5s for smoother playback
            if sleep_time > 0:
                time.sleep(sleep_time)
        last_ts = ts

    print("Streaming complete. View at https://app.rerun.io")
    print("Press Ctrl+C to exit...")

    # Keep the script running so you can view the data
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream labeled .pcd to rerun.io")
    parser.add_argument("pcd_folder", help="Folder containing timestamped .pcd files")
    parser.add_argument(
        "--start_index",
        "-s",
        type=int,
        default=0,
        help="Start frame index (0-based, not timestamp)",
    )
    args = parser.parse_args()
    main(args.pcd_folder, args.start_index)
