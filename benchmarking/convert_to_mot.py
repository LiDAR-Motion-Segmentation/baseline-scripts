# The standard format for MOT benchmarks is a CSV file with lines: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def convert_json_to_mot(json_dir, output_file):
    json_dir = Path(json_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_json_files = sorted(list(json_dir.glob("*.json")))

    valid_files = []

    # Filter: Only process files that are numbers (e.g., "000000.json")
    for f_path in all_json_files:
        if f_path.stem.isdigit():
            valid_files.append(f_path)

    if not valid_files:
        print(f"No numeric JSON files (e.g., 000000.json) found in {json_dir}")
        return

    print(f"Converting {len(valid_files)} frames from {json_dir}...")

    with open(output_file, "w") as f_out:
        for f_path in tqdm(valid_files):
            try:
                frame_idx = int(f_path.stem) + 1
            except ValueError:
                continue

            with open(f_path, "r") as f_in:
                try:
                    frame_data = json.load(f_in)
                except json.JSONDecodeError:
                    print(f"Skipping corrupt file: {f_path}")
                    continue

            for entity in frame_data:
                # Use GLOBAL ID for benchmarking
                track_id = entity.get("global_id", -1)

                # Filter out "lost" tracks or those without an ID
                if track_id == -1:
                    continue

                x1, y1, x2, y2 = entity["bbox"]
                w = x2 - x1
                h = y2 - y1
                conf = entity.get("confidence", 1.0)

                # Format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
                line = f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                f_out.write(line)

    print(f"Success! Saved MOT results to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Option A: Automatic path inference (assuming standard folder structure)
    parser.add_argument("--cam", type=int, help="Camera ID (to auto-find path)")
    parser.add_argument(
        "--base_dir", type=str, default="./MCMPT_output", help="Base output directory"
    )

    # Option B: Manual path override
    parser.add_argument(
        "--input_dir", type=str, help="Manually specify the JSON folder path"
    )
    parser.add_argument(
        "--output_file", type=str, help="Manually specify the output .txt file path"
    )

    args = parser.parse_args()

    if args.input_dir:
        json_path = Path(args.input_dir)
        if args.output_file:
            txt_path = Path(args.output_dir)
        else:
            txt_path = Path("./benchmarks") / f"{json_path.parent.name}.txt"
    elif args.cam:
        json_path = Path(args.base_dir) / f"cam_{args.cam}" / "json"
        txt_path = Path("./benchmarks") / f"cam_{args.cam}.txt"
    else:
        print("Error: Please provide either --cam or --input_dir")
        exit(1)

    convert_json_to_mot(json_path, txt_path)
