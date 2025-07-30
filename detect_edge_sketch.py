# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:35:06 2025

@author: yzhao
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


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
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_16410.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_17850.png"
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_18090.png" # no pupil
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_22380.png"


# ---- Load grayscale image ----
pupil_image = Path(DATA_PATH) / image_file
img = cv2.imread(pupil_image, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(3, 3))
# plt.subplot(5, 1, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")
# %%
# ---- Step 1: Gaussian Blur ----
blur_ksize = 5
img_blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

plt.figure(figsize=(3, 3))
# plt.subplot(5, 1, 1)
plt.imshow(img_blurred, cmap="gray")
plt.title("Original")
plt.axis("off")

# %%
# ---- Step 2: Canny Edge Detection ----
low_thresh, high_thresh = 30, 50
edges = cv2.Canny(img_blurred, low_thresh, high_thresh)

plt.figure(figsize=(3, 3))
plt.imshow(edges, cmap="gray")
plt.title("Canny Edges")
plt.axis("off")  # optional: hide axes
plt.show()
# %%
# ---- Step 3: Find Contours ----
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ---- Step 4: Evaluate Contours Based on Area and Darkness ----
min_area, max_area = 25, 7200
best_contour = None
lowest_mean_intensity = np.inf

# largest_cnt = max(contours, key=cv2.contourArea)
min_area = 25
max_distance = 30  # pixels
filtered_pts = []
debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        continue

    pt = tuple(map(float, cnt[0][0]))
    filtered_pts.extend(cnt)
    cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 1)

if not filtered_pts:
    print(f"No contours found with area > {min_area}.")
    convex_hull = None
else:
    all_pts = np.concatenate(filtered_pts, axis=0)  # shape (M, 1, 2)
    convex_hull = cv2.convexHull(all_pts)  # shape (H, 1, 2)

    overlay = debug_img.copy()
    cv2.polylines(overlay, [convex_hull], isClosed=True, color=(0, 0, 255), thickness=1)

    # Blend the overlay with the original image (fake transparency)
    alpha = 0.5  # transparency level (0 = only base, 1 = only overlay)
    cv2.addWeighted(overlay, alpha, debug_img, 1 - alpha, 0, debug_img)

plt.figure(figsize=(3, 3))
plt.imshow(debug_img)
plt.title("Contours + Convex Hull")
plt.axis("off")
