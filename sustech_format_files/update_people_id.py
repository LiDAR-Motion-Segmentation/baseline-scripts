#!/usr/bin/env python3
"""
Script to update obj_type for specific moving_people objects in JSON files
For moving_people objects:
- obj_id 1 → change obj_type to people_static
- obj_id 3 → change obj_type to people_static
- obj_id 2 → keep as moving_people (no change)
"""

import json
import os
import glob
import sys
from pathlib import Path


def update_people_obj_type(data, source_obj_type="moving_people"):
    """
    Update obj_type for objects matching specific criteria

    Args:
        data: List of objects from JSON file
        source_obj_type: Only update objects with this current obj_type

    Returns:
        Updated data with new obj_type values and list of changes made
    """
    changes_made = []

    # Define the mapping: obj_id -> new_obj_type
    id_to_type_mapping = {
        "1": "people_static",
        "3": "people_static",
        "2": "moving_people",  # This stays the same, but we'll mention it
    }

    for obj in data:
        if "obj_id" in obj and "obj_type" in obj:
            obj_id_str = str(obj["obj_id"])

            # Check if this object has the source obj_type
            if obj["obj_type"] == source_obj_type:
                if obj_id_str in id_to_type_mapping:
                    new_obj_type = id_to_type_mapping[obj_id_str]
                    old_obj_type = obj["obj_type"]

                    if old_obj_type != new_obj_type:
                        obj["obj_type"] = new_obj_type
                        change_info = f"  Changed obj_type from '{old_obj_type}' to '{new_obj_type}' for obj_id {obj['obj_id']}"
                        print(change_info)
                        changes_made.append(change_info)
                    else:
                        print(
                            f"  Found {source_obj_type} with obj_id {obj['obj_id']} (already correct type: {new_obj_type})"
                        )
                else:
                    print(
                        f"  Found {source_obj_type} with obj_id {obj['obj_id']} (no mapping defined)"
                    )
            # else:
            #     # Uncomment to see other object types
            #     print(f"  Skipped obj_type: {obj['obj_type']}, obj_id: {obj.get('obj_id', 'N/A')}")

    return data, changes_made


def process_json_files(directory_path, source_obj_type="moving_people", dry_run=False):
    """
    Process all JSON files in the directory

    Args:
        directory_path: Path to directory containing JSON files
        source_obj_type: Only update objects with this current obj_type
        dry_run: If True, don't save files, just show what would be changed
    """
    # Get all JSON files and sort them numerically
    json_files = glob.glob(os.path.join(directory_path, "*.json"))

    # Sort files numerically (assuming they have numeric names)
    try:
        json_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    except ValueError:
        # If files don't have numeric names, sort alphabetically
        json_files.sort()

    if not json_files:
        print(f"No JSON files found in directory: {directory_path}")
        return

    print(f"Found {len(json_files)} JSON files:")
    for i, file_path in enumerate(json_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")

    print(f"\nRules for '{source_obj_type}' objects:")
    print(f"  obj_id 1 → change obj_type to 'people_static'")
    print(f"  obj_id 3 → change obj_type to 'people_static'")
    print(f"  obj_id 2 → keep as 'moving_people' (no change)")
    print(f"Dry run: {dry_run}")
    print("-" * 50)

    total_changes = 0
    files_modified = 0

    # Process each file
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")

        try:
            # Read JSON file
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Update obj_type for specific objects
            updated_data, changes_made = update_people_obj_type(data, source_obj_type)

            if changes_made:
                files_modified += 1
                total_changes += len(changes_made)

                # Save updated file (if not dry run)
                if not dry_run:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(updated_data, f, indent=2)
                    print(f"  ✓ Updated and saved: {filename}")
                else:
                    print(f"  ✓ Would update: {filename}")
            else:
                print(f"  - No {source_obj_type} objects requiring changes found")

        except json.JSONDecodeError as e:
            print(f"  ✗ Error parsing JSON in {filename}: {e}")
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

    print(f"\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Files processed: {len(json_files)}")
    print(f"  Files modified: {files_modified}")
    print(f"  Total changes made: {total_changes}")


def main():
    # Configuration
    source_obj_type = "moving_people"

    # Get directory path from command line argument or use current directory
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = input(
            "Enter the directory path containing JSON files (or press Enter for current directory): "
        ).strip()
        if not directory_path:
            directory_path = "."

    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    # Display configuration
    print(f"Directory: {os.path.abspath(directory_path)}")
    print(f"Processing objects with obj_type: '{source_obj_type}'")
    print(f"\nRules:")
    print(f"  - obj_id 1 → change to 'people_static'")
    print(f"  - obj_id 3 → change to 'people_static'")
    print(f"  - obj_id 2 → keep as 'moving_people'")

    # Option to modify source obj_type
    modify = (
        input(f"\nDo you want to change the source obj_type? (y/n, default: n): ")
        .strip()
        .lower()
    )
    if modify == "y":
        source_obj_type = (
            input(f"Enter source obj_type (current: {source_obj_type}): ").strip()
            or source_obj_type
        )
        print(f"Updated to process objects with obj_type: '{source_obj_type}'")

    # Option for dry run
    dry_run_input = (
        input("\nDo you want to do a dry run first? (y/n, default: y): ")
        .strip()
        .lower()
    )
    dry_run = dry_run_input != "n"

    if dry_run:
        print("\n=== DRY RUN MODE ===")
        process_json_files(directory_path, source_obj_type, dry_run=True)

        proceed = (
            input(
                "\nDry run complete. Do you want to proceed with actual changes? (y/n): "
            )
            .strip()
            .lower()
        )
        if proceed != "y":
            print("Operation cancelled.")
            return

    # Process files
    print(f"\n=== PROCESSING FILES ===")
    process_json_files(directory_path, source_obj_type, dry_run=False)
    print("\n✓ All files processed successfully!")


if __name__ == "__main__":
    main()
