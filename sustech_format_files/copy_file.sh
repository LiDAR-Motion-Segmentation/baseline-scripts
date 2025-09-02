#!/bin/bash

# This script finds the last numbered JSON file in the 'label' directory,
# increments the number, and copies the last file to the new name.
# contributed by @Gaurav Kumar

TARGET_DIR="label"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' not found."
  exit 1
fi

# Get the last JSON file using a version sort to handle numbers correctly
last_file=$(ls -1 "$TARGET_DIR"/*.json | sort -V | tail -n 1)

if [ -z "$last_file" ]; then
  echo "Error: No .json files found in '$TARGET_DIR'."
  exit 1
fi

# Extract the number part of the filename (e.g., "000033")
last_number_str=$(basename "$last_file" .json)

# Safely convert to an integer and increment
next_number_int=$((10#$last_number_str + 1))

# Format the new number back to a 6-digit string (e.g., "000034")
next_number_str=$(printf "%06d" "$next_number_int")

# Define the full path for the new file
new_file="${TARGET_DIR}/${next_number_str}.json"

# Copy the last file to the new file
cp "$last_file" "$new_file"

echo "Created $new_file from $last_file"