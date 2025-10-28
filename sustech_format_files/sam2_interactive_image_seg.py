import numpy as np
import torch
import argparse
import os
import cv2
import time
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", help="path to the model checkpoints", required=True)
parser.add_argument("--input", help="path to the input image", required=True)
args = parser.parse_args()

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

clicked = []
labels = []
rectangles = []
mode = "point"  # using the deafualt point mode as of now
ix, iy = -1, -1
drawing = False
last_point_time = 0
delay = 0.2


def image_overlay(image, segmented_image):
    alpha = 0.6
    beta = 0.4
    gamma = 0
    segmented_image = np.array(segmented_image, dtype=np.float32)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image, dtype=np.float32) / 255
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image


def load_model(ckpt):
    # model_name = ckpt.split(os.path.sep)[-1]
    model_cfg = "sam2.1_l"
    model = build_sam2(model_cfg, ckpt, device="cuda", apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)
    return predictor


def get_mask(masks, random_color=False, border=True):
    for i, mask in enumerate(masks):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.float32)

        if i > 0:
            mask_image += mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        else:
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if border:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            mask_image = cv2.drawContours(
                mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
            )

    return mask_image


# mouse callback function needs to be added here
def draw(event, x, y, flags, param):
    global ix, iy, drawing, rectangles, clicked, labels, mode, last_point_time
    current_time = time.time()

    # show image parameter here will have to be changed
    if mode == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append([x, y])
            labels.append(1)
            cv2.circle(show_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("image", show_image)
        elif event == cv2.EVENT_MBUTTONDOWN:
            clicked.append([x, y])
            labels.append(0)
            cv2.circle(show_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("image", show_image)
        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                if current_time - last_point_time >= delay:
                    clicked.append([x, y])
                    labels.append(1)
                    cv2.circle(show_image, (x, y), 5, (0, 0, 255), -1)
                    cv2.imshow("image", show_image)
                    last_point_time = current_time

    elif mode == "rectangle":
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img = show_image.copy()
