#!/usr/bin/env python3

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import cv2

from tqdm import tqdm
from icecream import ic

def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_yaml(path):
    if yaml.__version__ >= '5.1':
        config = yaml.load(open(path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(path))
    return config

def load_image_files(image_folder):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file in sorted(os.listdir(image_folder)):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(os.path.join(image_folder, file))
    
    return image_files

def load_images_as_array(image_files):
    try:
        img = Image.open(image_path)

        # converting image to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # normalizing to [0, 1] range
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_default_config():
    return { 
        'num_frames': -1,
        'debug': False,
        'normalize': True,
        'num_last_n': 1,
        'visualize': False,
        'min_range': 0.0,
        'max_range': 1.0
    }

def process_image_sequence(config, input_dir, output_dir, output_suffix="residual"):
    # specify parameters
    num_frames = config['num_frames']
    debug = config['debug']
    normalize = config['normalize']
    num_last_n = config['num_last_n']
    visualize = config['visualize']

    # create output directory
    residual_image_folder = os.path.join(output_dir, f"images_{output_suffix}")

    # might crash need to check
    check_and_makedirs(residual_image_folder)

    if visualize:
        visualization_folder = os.path.join(output_dir, f"visualization_{output_suffix}")
        check_and_makedirs(residual_image_folder)

    # Load image files
    image_paths = load_image_files(input_dir)
    
    if not image_paths:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_paths)} images in {input_dir}")

    # limit number of frames if specified
    if num_frames > 0 and num_frames < len(image_paths):
        image_paths = image_paths[:num_frames]
        print(f"Processing first {num_frames} images")

    # process each image in sequence
    for frame_idx in tqdm(range(len(image_paths)), desc="Processing images"):
        file_name = os.path.join(residual_image_folder, f"{frame_idx:06d}")
        
        # load current image
        current_image = load_images_as_array(image_paths[frame_idx])
        if current_image is None:
            continue

        # for the first N frames, generate dummy residual (zeros)
        if frame_idx < num_last_n:
            diff_image = np.zeros_like(current_image, dtype=np.float32)
            np.save(file_name, diff_image)
            continue

        # Load previous image (num_last_n frames back)
        prev_frame_idx = frame_idx - num_last_n
        prev_image = load_images_as_array(image_paths[prev_frame_idx])

        if prev_image is None:
            diff_image = np.zeros_like(current_image, dtype=np.float32)
            np.save(file_name, diff_image)
            continue

        if current_image.shape != prev_image.shape:
            # resizing previous image to match the current image
            prev_image = cv2.resize(prev_image,
                                    (current_image.shape[1], current_image.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

        # generating residual image
        valid_mask = (current_image >= config['min_range']) & \
                        (current_image <= config['max_range']) & \
                        (prev_image >= config['min_range']) & \
                        (prev_image <= config['max_range'])
                        
        # calculate the absolute difference
        difference = np.abs(current_image - prev_image)

        # apply normalization if enabled
        if normalize:
            # avoid division by zero
            valid_current = current_image[valid_mask]
            valid_current[valid_current == 0] = 1e-8
            difference[valid_mask] = difference[valid_mask] / valid_current

        # create residual image
        diff_image = np.zeros_like(current_image, dtype=np.float32)
        diff_image[valid_mask] = difference[valid_mask]

        # for debugging visualization
        if debug:
            fig, axs = plt.subplots(3, figsize=(12, 8))
            axs[0].imshow(prev_image, cmap='gray')
            axs[0].set_title(f'Previous Image (frame {prev_frame_idx})')
            axs[1].imshow(current_image, cmap='gray')
            axs[1].set_title(f'Current Image (frame {frame_idx})')
            axs[2].imshow(diff_image, cmap='hot', vmin=0, vmax=1)
            axs[2].set_title('Residual Image')
            plt.tight_layout()
            plt.show()

        # Save visualization if enabled
        if visualize:
            fig = plt.figure(frameon=False, figsize=(16, 10))
            fig.set_size_inches(20.48, 0.64)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(diff_image, cmap='hot', vmin=0, vmax=1)
            image_name = os.path.join(visualization_folder, f"{frame_idx:06d}.png")
            plt.savefig(image_name, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        # Save residual image as numpy array
        np.save(file_name, diff_image)
    print(f"Residual image generation completed! Output saved to: {residual_image_folder}")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Generate residual images from image sequence')
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('output_dir', type=str, help='Output directory for residual images')
    parser.add_argument('--output_suffix', type=str, default='residual', 
                       help='Suffix for output directory name (default: residual)')
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to YAML configuration file')
    parser.add_argument('--num_last_n', type=int, default=1, 
                       help='Number of previous frames to use for residual calculation')
    parser.add_argument('--normalize', action='store_true', 
                       help='Enable normalization of residual values')
    parser.add_argument('--visualize', action='store_true', 
                       help='Save visualization images')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with matplotlib display')
    parser.add_argument('--num_frames', type=int, default=-1, 
                       help='Number of frames to process (-1 for all)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_yaml(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = create_default_config()
        print("Using default configuration")
    
    # Override config with command line arguments
    if args.num_last_n:
        config['num_last_n'] = args.num_last_n
    if args.normalize:
        config['normalize'] = args.normalize
    if args.visualize:
        config['visualize'] = args.visualize
    if args.debug:
        config['debug'] = args.debug
    if args.num_frames > 0:
        config['num_frames'] = args.num_frames
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    # Create output directory
    check_and_makedirs(args.output_dir)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output suffix: {args.output_suffix}")
    print(f"Configuration: {config}")
    print("-" * 50)
    
    # Process the image sequence
    process_image_sequence(config, args.input_dir, args.output_dir, args.output_suffix)

if __name__ == '__main__':
    main()