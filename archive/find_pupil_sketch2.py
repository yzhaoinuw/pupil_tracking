# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 22:18:11 2025

@author: yzhao
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass


def exponential_dark_affinity(img, threshold=50, decay_rate=0.2, percentile=0):
    """
    Create a normalized heatmap where darker pixels under the threshold have higher affinity.
    Darkest reference is based on a percentile. Pixels darker than it are capped at max affinity.
    """
    mask = img < threshold
    dark_pixels = img[mask]

    if dark_pixels.size == 0:
        return np.zeros_like(img, dtype=np.float32)

    darkest_val = np.percentile(dark_pixels, percentile)

    affinity = np.zeros_like(img, dtype=np.float32)
    delta = np.clip(img[mask] - darkest_val, a_min=0, a_max=None)
    affinity[mask] = np.exp(-decay_rate * delta)

    # Normalize to [0, 1]
    affinity -= affinity.min()
    if affinity.max() > 0:
        affinity /= affinity.max()

    return affinity


# %%
DATA_PATH = "./data_cropped_centered/"
# Load grayscale image
# can't find contour
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_3686.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_27210.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_11640.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_3880.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_1455.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_0485.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_20079.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_31913.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_42680.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_40546.png"
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_69064.png"

# low contrast pupil
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_67609.png"
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_82547.png"  # hard
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_82741.png"  # hard
image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_84196.png"  # hard

# small, subtle pupil
image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_0030.png"  # subtle pupil
image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_0450.png"  # subtle pupil

# large pupil
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_16260.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_16410.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_17850.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_22380.png"

# no pupil, eye closed
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_18090.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_7410.png"

# %%
pupil_image = Path(DATA_PATH) / image_file
# Example usage
img = cv2.imread(pupil_image, cv2.IMREAD_GRAYSCALE)
img_enhanced = exponential_dark_affinity(img, threshold=50, decay_rate=0.25, percentile=1)
com_y, com_x = center_of_mass(img_enhanced)  # Note: (y, x) order

plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray")
plt.title("Original Grayscale")
plt.axis("off")

plt.figure(figsize=(3, 3))
plt.imshow(img_enhanced, cmap="hot")
plt.scatter([com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass")
plt.title("Contrast Enhanced")
plt.axis("off")

# %%
# img_open = cv2.morphologyEx(img_enhanced, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
img_eroded = cv2.erode(img_enhanced, np.ones((3, 3), np.uint8))

com_y, com_x = center_of_mass(img_eroded)  # Note: (y, x) order

plt.figure(figsize=(3, 3))
plt.imshow(img_eroded, cmap="hot")
plt.scatter([com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass")
plt.title("Eroded")
plt.axis("off")

# %%
img_smoothed = cv2.GaussianBlur(img_eroded, (5, 5), 0)
com_y, com_x = center_of_mass(img_smoothed)  # Note: (y, x) order

plt.figure(figsize=(3, 3))
plt.imshow(img_smoothed, cmap="hot")
plt.scatter([com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass")
plt.title("Smoothed")
plt.axis("off")

# %%
"""
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
img_closed = cv2.morphologyEx(img_smoothed, cv2.MORPH_CLOSE, kernel)
com_y, com_x = center_of_mass(img_closed)  # Note: (y, x) order

plt.figure(figsize=(3, 3))
plt.imshow(img_closed, cmap='hot')
plt.scatter([com_x], [com_y], color='cyan', marker='+', s=80, label="Center of Mass")
plt.title("Closed")
plt.axis("off")
"""
# %%
# Step 3: Thresholding by percentile
img_smoothed = np.clip(img_smoothed * 255, 0, 255).astype(np.uint8)
nonzero_vals = img_smoothed[img_smoothed > 0]
thresh_val = np.percentile(nonzero_vals, 50)
img_threshed = np.where(img_smoothed > thresh_val, img_smoothed, 0)

plt.figure(figsize=(3, 3))
plt.imshow(img_threshed / 255, cmap="hot")
plt.scatter([com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass")
plt.title("Threshed")
plt.axis("off")


# %%

contours, _ = cv2.findContours(img_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# largest_cnt = max(contours, key=cv2.contourArea)
min_area = 25
max_distance = 30  # pixels
filtered_pts = []
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
com_y, com_x = center_of_mass(img_threshed)  # Note: (y, x) order

for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        continue

    filtered_pts.extend(cnt)
    cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), 1)

plt.figure(figsize=(3, 3))
plt.imshow(img_contours)
plt.scatter([com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass")
plt.title("Contours")
plt.axis("off")
