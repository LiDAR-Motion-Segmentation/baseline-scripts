#!/usr/bin/env python3

"""
Script to batch-update the 'obj_type' field in JSON annotations for people types.
Usage:
    python batch_update_obj_type.py --json_dir /path/to/jsons --mode forward
Modes:
  - forward:  people.moving → moving_people,   people.static → people_static
  - reverse:  moving_people → people.moving,   people_static → people.static
"""

import os
import argparse
import json
from pathlib import Path
from typing import List


def update_obj_types_in_json(filepath: Path, mode: str) -> bool:
    with open(filepath, "r") as f:
        data = json.load(f)

    updated = False
    if mode == "forward":
        mapping = {"people.moving": "moving_people", "people.static": "people_static"}
    elif mode == "reverse":
        mapping = {"moving_people": "people.moving", "people_static": "people.static"}
    else:
        raise ValueError("Mode must be either forward or reverse")

    for obj in data:
        old_type = obj.get("obj_type", None)
        if old_type in mapping:
            obj["obj_type"] = mapping[old_type]
            updated = True

    if updated:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    return updated


def process_directory(json_dir: Path, mode: str) -> List[Path]:
    updated_files = []
    for json_path in sorted(json_dir.glob("*.json")):
        changed = update_obj_types_in_json(json_path, mode)
        if changed:
            updated_files.append(json_path)
    print(f"Total files updated: {len(updated_files)}")
    return updated_files


def main():
    parser = argparse.ArgumentParser(
        description="Batch update obj_type in annotation json"
    )
    parser.add_argument(
        "--json_dir",
        required=True,
        type=str,
        help="Directory containing annotation JSON files.",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "reverse"],
        help="Which mapping to perform; forward=people.moving→moving_people, reverse=vice versa.",
    )
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.is_dir():
        raise ValueError(f"input directory does not exist: {json_dir}")

    process_directory(json_dir, args.mode)


if __name__ == "__main__":
    main()
