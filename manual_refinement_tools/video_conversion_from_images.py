import cv2
import os
import argparse
import re

def natural_sort_key(s):
    """
    Splits string into segments of text and numbers for natural sorting.
    This ensures 'img_2.png' comes before 'img_10.png', 
    and handles '000000.png' sequences correctly.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_video_from_images(image_folder, output_file, fps):
    if not os.path.exists(image_folder):
        print(f"Error: Directory '{image_folder}' not found.")
        return

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # 1. Gather all images
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(valid_extensions)]
    
    # 2. Sort naturally (Handles 000000 -> 000001 -> ... -> 000010 correctly)
    images.sort(key=natural_sort_key)

    if not images:
        print(f"No images found in '{image_folder}'.")
        return

    # 3. Print verification info
    print(f"Found {len(images)} images.")
    print(f"Sequence starts with: {images[0]}")
    print(f"Sequence ends with:   {images[-1]}")

    # 4. Setup Video Writer based on the first frame
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 5. Process
    print(f"Writing video to {output_file}...")
    for i, image_name in enumerate(images):
        path = os.path.join(image_folder, image_name)
        img = cv2.imread(path)

        # Safety check for resizing
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))

        out.write(img)
        
        # Optional: Print progress every 100 frames
        if i % 100 == 0:
            print(f"Processed {i}/{len(images)} frames...", end='\r')

    out.release()
    print(f"\nDone! Video saved as '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to images (e.g., ./dataset/sequence_1)")
    parser.add_argument("-o", "--output", default="output.mp4", help="Output filename")
    parser.add_argument("-fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    create_video_from_images(args.path, args.output, args.fps)