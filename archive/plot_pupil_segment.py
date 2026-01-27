# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:16:26 2025

@author: yzhao
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load image
image_file = "C:/Users/yzhao/python_projects/pupil_tracking/data_cropped_centered/250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_0630.png"
img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

# Thresholding to isolate dark blobs
_, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# Find contours and extract the largest blob
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)

# Create mask of the largest blob
mask = np.zeros_like(img)
cv2.drawContours(mask, [largest], -1, 255, -1)

# Convert original grayscale image to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# Create red overlay where mask is 255
overlay = img_rgb.copy()
overlay[mask == 255] = [255, 0, 0]  # Red

# Blend original and overlay
alpha = 0.5
blended = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(blended)
plt.title("Pupil Mask Overlay")
plt.axis("off")
plt.tight_layout()
plt.show()
