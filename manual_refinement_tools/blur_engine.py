import cv2
import numpy as np
from typing import Tuple


class BlurEngine:
    """
    Handles the mathematical application of Gaussian blur with
    feathered circular masking.
    """

    def __init__(self, intensity: float, feather_ratio: float):
        """
        Args:
            intensity: Strength of blur relative to ROI size (0.0 - 1.0).
            feather_ratio: Softness of edges relative to radius (0.0 - 0.5).
        """
        self.intensity = intensity
        self.feather_ratio = feather_ratio

    def apply_round_blur(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ):

        # safe ROI extraction with boundary checks
        h_img, w_img = image.shape[:2]
        pad = int(min(w, h) * 0.1)
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return image

        roi_h, roi_w = roi.shape[:2]

        # dynamic kernel calculation
        ksize = int(min(roi_h, roi_w) * self.intensity)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        ksize = max(3, ksize)

        # gaussian filter
        blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)

        # circular mask
        mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        center = (roi_w // 2, roi_h // 2)
        radius = min(roi_w, roi_h) // 2
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        # edge softening
        mask_blur_k = int(radius * self.feather_ratio) | 1
        mask_soft = cv2.GaussianBlur(mask, (mask_blur_k, mask_blur_k), 0)

        # alpha blending
        mask_norm = mask_soft.astype(float) / 255.0
        mask_norm = cv2.merge([mask_norm, mask_norm, mask_norm])
        result_roi = (blurred_roi.astype(float) * mask_norm) + (
            roi.astype(float) * (1.0 - mask_norm)
        )
        image[y1:y2, x1:x2] = result_roi.astype(np.uint)
        return image
