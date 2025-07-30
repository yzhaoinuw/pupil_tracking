# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 20:31:16 2025

@author: yzhao
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import remove_appendages, enhance_contrast, draw_mask_contour

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


pupil_range_contour_file = Path("./") / "pupil_ROI.npy"
pupil_image = Path(DATA_PATH) / image_file
# estimate_pupil_info(pupil_image, pupil_range_contour_path, debug_plot_save_dir="./")

img = cv2.imread(pupil_image, cv2.IMREAD_GRAYSCALE)

# Plot side-by-side
plt.figure(figsize=(3, 3))
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

# %% load ROI
ROI = np.load(pupil_range_contour_file)
mask = np.zeros_like(img, dtype=np.uint8)
cv2.fillPoly(mask, [ROI], 255)

# Create a white image
white_background = np.full_like(img, 255, dtype=np.uint8)

# img = cv2.GaussianBlur(img, (3, 15), 0)
# Combine: keep ROI region, fill rest with white
ROI_img = np.where(mask == 255, img, 255).astype(np.uint8)

# %%
img_enhanced = enhance_contrast(ROI_img, dark_threshold=20)
plt.figure(figsize=(3, 3))
plt.imshow(img_enhanced, cmap="gray")
plt.title("First Enhancement")
plt.axis("off")

# %%
img_blurred = cv2.GaussianBlur(img_enhanced, (3, 21), 0)
img_enhanced = enhance_contrast(img_blurred, dark_threshold=30)

plt.figure(figsize=(3, 3))
plt.imshow(img_enhanced, cmap="gray")
plt.title("Pupil ROI Enhanced")
plt.axis("off")


# %%
_, img_threshed = cv2.threshold(img_enhanced, 30, 255, cv2.THRESH_BINARY_INV)

# --- Step 3: Morphological cleaning (on masked region only) ---
img_morphed = cv2.morphologyEx(img_threshed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
img_morphed = cv2.morphologyEx(img_morphed, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(3, 3))
plt.imshow(img_morphed, cmap="gray")
plt.title("Smoothed and Morphed")
plt.axis("off")

# %%
# img_morphed = remove_appendages(img_morphed)
# --- Step 4: Find contours ---
contours, _ = cv2.findContours(img_morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_cnt = max(contours, key=cv2.contourArea)
min_area = 25
max_distance = 30  # pixels
filtered_pts = []
debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    if cv2.contourArea(cnt) < min_area:
        continue

    pt = tuple(map(float, cnt[0][0]))
    dist = cv2.pointPolygonTest(largest_cnt, pt, True)
    if abs(dist) <= max_distance:
        filtered_pts.extend(cnt)
        cv2.drawContours(debug_img, [cnt], -1, (0, 255, 0), 1)

if not filtered_pts:
    print("No contours found with area > 5.")
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

# %%
# Fit ellipse to the largest contour
ellipse = cv2.fitEllipse(convex_hull)

# Fit a circle to the largest contour
(center_x, center_y), radius = cv2.minEnclosingCircle(convex_hull)
center = (int(center_x), int(center_y))
radius = int(radius)

ellipse_center, axes_len, angle = ellipse
adjusted_angle = -(angle - 90)
diameter = 2 * radius
info = {
    "center_circle": center,
    "diameter": diameter,
    "ellipse_center": ellipse_center,
    "ellipse_a1": max(axes_len),
    "ellipse_a2": min(axes_len),
    "ellipse_angle": adjusted_angle,
}

# %% plotting
# Overlay

img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
blended = img_rgb.copy()

# Draw the ellipse on a blank mask
ellipse_mask = np.zeros_like(img)
cv2.ellipse(ellipse_mask, ellipse, 255, -1)

# Draw the circle on a blank mask
circle_mask = np.zeros_like(img)
cv2.circle(circle_mask, center, radius, 255, -1)

# Draw both contours on the same overlay
blended = draw_mask_contour(blended, ellipse_mask, color=(255, 0, 0), thickness=1)
blended = draw_mask_contour(blended, circle_mask, color=(0, 0, 255), thickness=1)

plt.figure(figsize=(3, 3))
plt.imshow(blended)
plt.title("Labeled")
plt.axis("off")

plt.tight_layout()
# fig.savefig(save_path, dpi=200)
# plt.close(fig)
plt.show()
