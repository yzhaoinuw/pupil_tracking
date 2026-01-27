# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 13:03:41 2025

@author: yzhao
"""


import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def remove_appendages(binary_mask, max_kernel_frac=0.2):
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
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary_mask

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    d = int(max(w, h) * max_kernel_frac)
    d = max(3, d | 1)  # Ensure it's odd and not too small

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask


def hybrid_contrast_enhance(img, low_percentile=5, gamma=0.4, clip_limit=2.0, tile_size=(5, 5)):
    """
    Hybrid contrast enhancement:
    - Globally crush darkest pixels using percentile + gamma
    - Then locally boost contrast using CLAHE

    Parameters:
    - img: Grayscale uint8 image
    - low_percentile: Intensity percentile to treat as new black
    - gamma: < 1 for aggressive dark compression
    - clip_limit: CLAHE contrast limit
    - tile_size: CLAHE tile grid size

    Returns:
    - Enhanced image (uint8)
    """
    # Step 1: Global dark push
    img = img.astype(np.float32)
    p = np.percentile(img, low_percentile)
    img_shifted = np.clip(img - p, 0, None)
    img_scaled = img_shifted / (255 - p)
    img_gamma = np.power(img_scaled, gamma)
    img_boosted = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)

    # Step 2: Local CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    img_final = clahe.apply(img_boosted)

    return img_final


DATA_PATH = "./data_cropped_centered/"
# Load grayscale image
# can't find contour
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_3686.png"

image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_27210.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_11640.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_3880.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_1455.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_0485.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_20079.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_31913.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_42680.png"


img = cv2.imread(os.path.join(DATA_PATH, image_file), cv2.IMREAD_GRAYSCALE)

# Step 1: Enhance contrast
img_eq = hybrid_contrast_enhance(img)
# Threshold and clean
# _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
_, thresh = cv2.threshold(img_eq, 40, 255, cv2.THRESH_BINARY_INV)

# Optional: blur to smooth small edges
thresh = cv2.GaussianBlur(thresh, (5, 5), 0)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
# thresh_filled = fill_pupil_discontinuities(thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# thresh = remove_appendages(thresh)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

# Fit ellipse to the largest contour
ellipse = cv2.fitEllipse(largest)
# Draw ellipse on a blank mask
mask = np.zeros_like(img)
cv2.ellipse(mask, ellipse, 255, -1)

"""
#
# Fit a circle to the largest contour
(center_x, center_y), radius = cv2.minEnclosingCircle(largest)
center = (int(center_x), int(center_y))
radius = int(radius)
# Draw circle on a blank mask
mask = np.zeros_like(img)
cv2.circle(mask, center, radius, 255, -1)
"""

# Overlay
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
overlay = img_rgb.copy()
overlay[mask == 255] = [255, 0, 0]  # Red

alpha = 0.15
blended = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)

# Plot side-by-side
plt.figure(figsize=(3, 12))

plt.subplot(4, 1, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(4, 1, 2)
plt.imshow(img_eq, cmap="gray")
plt.title("CLAHE Enhanced Image")
plt.axis("off")

plt.subplot(4, 1, 3)
plt.imshow(thresh, cmap="gray")
plt.title("thresh")
plt.axis("off")

plt.subplot(4, 1, 4)
plt.imshow(blended)
plt.title("Smoothed Pupil Mask (Ellipse Overlay)")
plt.axis("off")

plt.tight_layout()
plt.show()
