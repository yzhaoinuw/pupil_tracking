# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:12:49 2025

@author: yzhao
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def nonlinear_dark_scaling(img, threshold=50, gamma=2.5):
    mask = img < threshold
    scaled = np.zeros_like(img, dtype=np.float32)
    scaled[mask] = (threshold - img[mask]) ** gamma
    return scaled


def gaussian_2d(coords, amp, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords
    inner = ((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2)
    return amp * np.exp(-inner) + offset


def fit_gaussian(img_scaled):
    h, w = img_scaled.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = (x.ravel(), y.ravel())
    data = img_scaled.ravel()

    # Initial guess: amplitude, x0, y0, sigma_x, sigma_y, offset
    guess = (np.max(data), w // 2, h // 2, w // 6, h // 6, np.min(data))
    popt, _ = curve_fit(gaussian_2d, coords, data, p0=guess, maxfev=5000)
    return popt


def estimate_pupil_center(img_gray):
    scaled = nonlinear_dark_scaling(img_gray, threshold=50, gamma=3)
    params = fit_gaussian(scaled)
    amp, x0, y0, sigma_x, sigma_y, offset = params
    return (x0, y0), scaled


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

# problematic
image_file = (
    "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_67609.png"
)
image_file = (
    "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_82547.png"  # hard
)
image_file = (
    "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_82741.png"  # hard
)
image_file = (
    "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_84196.png"  # hard
)
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_0030.png" # subtle pupil
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_0450.png" # subtle pupil
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_7410.png" # no pupil, eye closed
image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_16260.png"
image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_16410.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_17850.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_18090.png" # no pupil
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_22380.png"


pupil_image = Path(DATA_PATH) / image_file
# Example usage:
img = cv2.imread(pupil_image, cv2.IMREAD_GRAYSCALE)

# Plot side-by-side
plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")
# %%
center, weighted_map = estimate_pupil_center(img)
print(f"Pupil center estimate: {center}")

# Visualization
plt.figure(figsize=(3, 3))
plt.imshow(weighted_map, cmap="hot")
plt.scatter([center[0]], [center[1]], color="cyan", marker="+")
plt.title("Weighted Darkness")
plt.axis("off")
