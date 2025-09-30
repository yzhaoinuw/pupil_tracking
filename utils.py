# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:16:55 2025

@author: yzhao
"""

import cv2
import numpy as np

from scipy.ndimage import center_of_mass


def expose_hot_area(img, threshold=50, max_threshold=50, decay_rate=0.2, percentile=1):
    """
    Create a normalized heatmap where darker pixels under the threshold have higher affinity.
    Darkest reference is based on a percentile. Pixels darker than it are capped at max affinity.
    """
    dark_pixels = np.array([])
    while dark_pixels.size == 0 and threshold <= max_threshold:
        mask = img < threshold
        dark_pixels = img[mask]
        threshold += 10

    if dark_pixels.size == 0:
        return np.zeros_like(img, dtype=np.float32)

    darkest_val = np.percentile(dark_pixels, percentile)
    affinity = np.zeros_like(img, dtype=np.float32)
    delta = np.clip(img[mask] - darkest_val, a_min=0, a_max=None)
    affinity[mask] = np.exp(-decay_rate * delta)

    # Normalize to [0, 1]
    # affinity -= affinity.min()
    # affinity /= affinity.max()
    return affinity


def gaussian_mask(size=150, sigma_x=30, sigma_y=15, x0=None, y0=None):
    """
    Create a 2D elliptical Gaussian mask centered at (x0, y0).

    Parameters:
    - size: width and height of the square mask
    - sigma_x: standard deviation along x-axis (controls width)
    - sigma_y: standard deviation along y-axis (controls height)
    - x0, y0: center coordinates of the Gaussian (in pixel indices).
              Defaults to center of the image.

    Returns:
    - mask: 2D numpy array with values in [0, 1]
    """
    if x0 is None:
        x0 = size // 2
    if y0 is None:
        y0 = size // 2

    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    squared_distance = ((xx - x0) ** 2) / (2 * sigma_x**2) + ((yy - y0) ** 2) / (
        2 * sigma_y**2
    )
    mask = np.exp(-squared_distance)
    return mask


def get_contour_image(contour, image):
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
    contour_image = image * mask
    return contour_image


def get_center_of_mass(image):
    y, x = center_of_mass(image)  # Note: (y, x) order
    return np.array([x, y])


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
