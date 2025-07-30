# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:24:36 2025

@author: yzhao
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = "./data_cropped_centered/"
# Load grayscale image
image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_27210.png"
image_file = (
    "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_11640.png"
)
img = cv2.imread(os.path.join(DATA_PATH, image_file), cv2.IMREAD_GRAYSCALE)

# Step 1: Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_eq = clahe.apply(img)
# Threshold and clean
# _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
_, thresh = cv2.threshold(img_eq, 50, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
# thresh_filled = fill_pupil_discontinuities(thresh)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Optional: blur to smooth small edges
thresh_blur = cv2.GaussianBlur(thresh, (5, 5), 0)

# Find contours
contours, _ = cv2.findContours(thresh_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

# Fit ellipse to the largest contour
ellipse = cv2.fitEllipse(largest)

# Draw ellipse on a blank mask
mask = np.zeros_like(img)
cv2.ellipse(mask, ellipse, 255, -1)

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
