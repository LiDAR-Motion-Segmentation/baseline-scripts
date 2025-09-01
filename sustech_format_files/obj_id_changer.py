#!/usr/bin/env python3
"""
Script to update obj_id values in JSON files according to specified mapping
Mapping: 2 -> 3, 5 -> 4, 3 -> 1, 6 -> 2
"""

import json
import os
import glob
import sys
from pathlib import Path

def update_obj_ids(data, id_mapping):
    """
    Update obj_id values in the data according to the mapping
    
    Args:
        data: List of objects from JSON file
        id_mapping: Dictionary mapping old IDs to new IDs
    
    Returns:
        Updated data with new obj_id values
    """
    for obj in data:
        if 'obj_id' in obj:
            old_id = obj['obj_id']
            # Convert to string for comparison (in case IDs are stored as strings)
            old_id_str = str(old_id)
            
            if old_id_str in id_mapping:
                obj['obj_id'] = id_mapping[old_id_str]
                print(f"  Changed obj_id from {old_id} to {obj['obj_id']}")
    
    return data

def process_json_files(directory_path, id_mapping, dry_run=False):
    """
    Process all JSON files in the directory
    
    Args:
        directory_path: Path to directory containing JSON files
        id_mapping: Dictionary mapping old IDs to new IDs
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
    
    print(f"\nID Mapping: {id_mapping}")
    print(f"Dry run: {dry_run}")
    print("-" * 50)
    
    # Process each file
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        try:
            # Read JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update obj_ids
            original_data = json.dumps(data, indent=2)
            updated_data = update_obj_ids(data, id_mapping)
            
            # Save updated file (if not dry run)
            if not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(updated_data, f, indent=2)
                print(f"  ✓ Updated and saved: {filename}")
            else:
                print(f"  ✓ Would update: {filename}")
                
        except json.JSONDecodeError as e:
            print(f"  ✗ Error parsing JSON in {filename}: {e}")
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")

def main():
    # Define the ID mapping: old_id -> new_id
    id_mapping = {
        "2": "3",  # 2 -> 3
        "5": "4",  # 5 -> 4  
        "3": "1",  # 3 -> 1
        "6": "2"   # 6 -> 2
    }
    
    # Get directory path from command line argument or use current directory
    if len(sys.argv) > 1:
        directory_path = sys.argv[1]
    else:
        directory_path = input("Enter the directory path containing JSON files (or press Enter for current directory): ").strip()
        if not directory_path:
            directory_path = "."
    
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    
    # Ask for confirmation
    print(f"Directory: {os.path.abspath(directory_path)}")
    print(f"ID Mapping: {id_mapping}")
    
    # Option for dry run
    dry_run_input = input("\nDo you want to do a dry run first? (y/n, default: y): ").strip().lower()
    dry_run = dry_run_input != 'n'
    
    if dry_run:
        print("\n=== DRY RUN MODE ===")
        process_json_files(directory_path, id_mapping, dry_run=True)
        
        proceed = input("\nDry run complete. Do you want to proceed with actual changes? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Operation cancelled.")
            return
    
    # Process files
    print(f"\n=== PROCESSING FILES ===")
    process_json_files(directory_path, id_mapping, dry_run=False)
    print("\n✓ All files processed successfully!")

if __name__ == "__main__":
    main()