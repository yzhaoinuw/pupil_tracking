# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 14:35:10 2025

@author: yzhao
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass
from utils import draw_mask_contour, expose_hot_area, get_mass_center

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
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_82547.png" # hard
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_82741.png" # hard
# image_file = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_84196.png" # hard

# small, subtle pupil
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_0030.png" # subtle pupil
# image_file = "250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701_0450.png" # subtle pupil

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
img_enhanced = expose_hot_area(img, threshold=50, decay_rate=0.25, percentile=1)
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
img_smoothed = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
com_y, com_x = center_of_mass(img_smoothed)  # Note: (y, x) order

plt.figure(figsize=(3, 3))
plt.imshow(img_smoothed, cmap="hot")
plt.scatter([com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass")
plt.title("Smoothed")
plt.axis("off")
# %%
# img_open = cv2.morphologyEx(img_smoothed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
img_eroded = cv2.erode(img_smoothed, np.ones((3, 3), np.uint8))

com_y, com_x = center_of_mass(img_eroded)  # Note: (y, x) order

plt.figure(figsize=(3, 3))
plt.imshow(img_eroded, cmap="hot")
plt.scatter([com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass")
plt.title("Eroded")
plt.axis("off")

# %%
"""
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
img_closed = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel)
com_y, com_x = center_of_mass(img_closed)  # Note: (y, x) order

plt.figure(figsize=(3, 3))
plt.imshow(img_closed, cmap='hot')
plt.scatter([com_x], [com_y], color='cyan', marker='+', s=80, label="Center of Mass")
plt.title("Closed")
plt.axis("off")
"""
# %%
# Step 3: Thresholding by percentile
img_eroded = np.clip(img_eroded * 255, 0, 255).astype(np.uint8)
# nonzero_vals = np.unique(img_eroded[img_eroded > 0])
nonzero_vals = img_eroded[img_eroded > 0]
# thresh_val = np.percentile(nonzero_vals, 25)

high_value = np.percentile(nonzero_vals, 95)
thresh_val = high_value / 20

img_threshed = np.where(img_eroded > thresh_val, img_eroded, 0)
# _, img_threshed = cv2.threshold(img_eroded, thresh_val, 255, cv2.THRESH_BINARY)
com_y, com_x = center_of_mass(img_threshed)  # Note: (y, x) order

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
filtered_contours = []
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
com_y, com_x = center_of_mass(img_threshed)  # Note: (y, x) order

for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        continue

    filtered_contours.append(cnt)
    cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), 1)

plt.figure(figsize=(3, 3))
plt.imshow(img_contours)
plt.title("Contours")
plt.axis("off")

# %%
image_center = (int(com_x), int(com_y))
contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
main_contour = None
for cnt in contours:
    dist = np.linalg.norm(get_mass_center(cnt) - image_center)
    if cv2.pointPolygonTest(cnt, image_center, measureDist=False) >= 0 or abs(dist) <= 20:
        main_contour = cnt
        main_contour_center = get_mass_center(cnt)
        break

# Step 3: Get size of main contour
x, y, w_main, h_main = cv2.boundingRect(main_contour)
max_main_dim = 2 * max(w_main, h_main)

img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# Step 4: filter based on width/height and distance from main
filtered = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    if w >= 3 * h or h >= 3 * w:
        continue  # Rule 1: too flat

    # Rule 2: too far from main contour
    dist = np.linalg.norm(get_mass_center(cnt) - main_contour_center)
    if dist > max_main_dim:
        continue

    filtered.append(cnt)
    cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), 1)

plt.figure(figsize=(3, 3))
plt.imshow(img_contours)
plt.title("Filtered Contours")
plt.axis("off")
# %%
# Combine all contour points into one array
all_pts = np.concatenate(filtered, axis=0)

# Fit ellipse to the largest contour
ellipse = cv2.fitEllipse(all_pts)

# Fit a circle to the largest contour
(center_x, center_y), radius = cv2.minEnclosingCircle(all_pts)
center = (int(center_x), int(center_y))
radius = int(radius)

ellipse_center, axes_len, angle = ellipse
adjusted_angle = -(angle - 90)
diameter = 2 * radius
# Check point count
if all_pts.shape[0] >= 5:
    ellipse = cv2.fitEllipse(all_pts)
    cv2.ellipse(img_contours, ellipse, (255, 0, 0), 2)

img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
overlay = img_rgb.copy()

# Draw the ellipse on a blank mask
ellipse_mask = np.zeros_like(img)
cv2.ellipse(ellipse_mask, ellipse, 255, -1)

# Draw the circle on a blank mask
circle_mask = np.zeros_like(img)
cv2.circle(circle_mask, center, radius, 255, -1)

blended = draw_mask_contour(overlay, ellipse_mask, color=(255, 0, 0), thickness=1)
blended = draw_mask_contour(overlay, circle_mask, color=(0, 0, 255), thickness=1)

fig = plt.figure(figsize=(3, 3))
plt.imshow(blended)
plt.title("Labeled")
plt.axis("off")
