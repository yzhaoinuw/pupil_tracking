# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 17:04:52 2025

@author: yzhao
"""


import cv2
import numpy as np


def remove_appendages(binary_mask, max_kernel_frac=0.15):
    """
    Remove thin appendages from a binary mask using morphological opening.
    The kernel size is scaled based on the size of the largest blob to avoid wiping out small pupils.

    Parameters:
    -----------
    binary_mask : np.ndarray
        Binary image (white = foreground/pupil, black = background).
    max_kernel_frac : float
        Maximum fraction of the largest contour diameter to use for kernel size.

    Returns:
    --------
    cleaned_mask : np.ndarray
        Mask with appendages removed but pupil preserved.
    """
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return binary_mask

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    d = int(max(w, h) * max_kernel_frac)
    d = max(3, d | 1)  # Ensure it's odd and not too small

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask


def enhance_contrast(
    img,
    dark_threshold=30,
    # relative_factor=10,
    gamma=2,
):

    img = img.astype(np.float32)

    # Define boost mask
    boost_mask = img <= dark_threshold

    if not np.any(boost_mask):
        return img.astype(np.uint8)  # No boosting needed

    # Normalize to [0, 1]
    scaled = img / 255.0

    # Create output image (copy original)
    corrected = scaled.copy()

    # Apply gamma darkening only to dark pixels
    corrected[boost_mask] = np.power(scaled[boost_mask], gamma)

    # Scale back to [0, 255]
    img_boosted = np.clip(corrected * 255, 0, 255).astype(np.uint8)

    return img_boosted


def draw_mask_contour(img, mask, color=(0, 0, 255), thickness=1):
    """
    Draw the contour of a binary mask on an image.

    Parameters:
    - img: RGB image to draw on (in-place)
    - mask: binary image (white = shape)
    - color: BGR color tuple
    - thickness: thickness of the contour line
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img
