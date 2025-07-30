# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:10:38 2025

@author: yzhao
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    expose_hot_area,
    get_contour_image,
    get_center_of_mass,
    draw_mask_contour,
)


def find_pupil_info(pupil_image, plot_save_dir="", debug_plot_save_dir=""):
    info = {
        "center_circle": None,
        "diameter": None,
        "ellipse_center": None,
        "ellipse_a1": None,
        "ellipse_a2": None,
        "ellipse_angle": None,
    }
    img = cv2.imread(pupil_image, cv2.IMREAD_GRAYSCALE)
    img_enhanced = expose_hot_area(img, threshold=50, decay_rate=0.25, percentile=1)
    img_smoothed = cv2.GaussianBlur(img_enhanced, (5, 5), 0)
    img_eroded = cv2.erode(img_smoothed, np.ones((3, 3), np.uint8))

    # Thresholding by percentile
    img_eroded = np.clip(img_eroded * 255, 0, 255).astype(np.uint8)
    nonzero_vals = img_eroded[img_eroded > 0]
    if nonzero_vals.size == 0:
        return info

    high_value = np.percentile(nonzero_vals, 95)
    thresh_val = high_value / 20
    img_threshed = np.where(img_eroded > thresh_val, img_eroded, 0)
    estimated_pupil_center = get_center_of_mass(img_threshed)

    # sort contours by their moment wrt estimated_pupil_center
    contours, _ = cv2.findContours(
        img_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_data = []
    min_area = 25
    img_significant_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        contour_image = get_contour_image(contour, img)
        mass = np.sum(contour_image)

        # Step 3: center of mass (intensity-weighted)
        com_contour_image = get_center_of_mass(contour_image)

        # Step 4: distance to estimated pupil center
        dist = np.linalg.norm(com_contour_image - estimated_pupil_center)

        # Step 5: mass × distance
        score = mass / dist
        contour_data.append((score, contour, com_contour_image))
        cv2.drawContours(img_significant_contours, [contour], -1, (0, 255, 0), 1)

    if not contour_data:
        return info

    # sort by descending score
    contour_data = sorted(contour_data, key=lambda x: x[0], reverse=True)

    # Find main contour and remove tiny contours
    _, main_contour, com_main_contour_image = contour_data[0]
    _, _, w_main, h_main = cv2.boundingRect(main_contour)
    max_main_dim = 2 * max(w_main, h_main)

    img_filtered_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Step 4: filter based on width/height and distance from main
    filtered_contours = [main_contour]
    cv2.drawContours(img_filtered_contours, [main_contour], -1, (0, 255, 0), 1)
    for _, contour, com_contour_image in contour_data[1:]:
        _, _, w, h = cv2.boundingRect(contour)

        # Rule 1: too flat
        if w >= 3 * h or h >= 3 * w:
            continue

        # Rule 2: too far from main contour
        dist = np.linalg.norm(com_contour_image - com_main_contour_image)
        if dist > max_main_dim:
            continue

        filtered_contours.append(contour)
        cv2.drawContours(img_filtered_contours, [contour], -1, (0, 255, 0), 1)

    # %% Pupil fitting with ellipse and circle
    # Combine all contour points into one array
    pupil_segment = np.concatenate(filtered_contours, axis=0)
    pupil_segment = cv2.convexHull(pupil_segment)

    # Fit ellipse to the largest contour
    ellipse = cv2.fitEllipse(pupil_segment)
    ellipse_center, axes_len, angle = ellipse
    ellipse_center = tuple(map(lambda v: int(round(v)), ellipse_center))
    axes_len = tuple(map(lambda v: int(round(v)), axes_len))
    adjusted_angle = -(int(round(angle)) - 90)

    # Fit a circle to the largest contour
    (center_x, center_y), radius = cv2.minEnclosingCircle(pupil_segment)
    radius = int(radius)  # cast as int for drawing purpose
    center = (int(center_x), int(center_y))
    diameter = 2 * radius

    info = {
        "center_circle": center,
        "diameter": diameter,
        "ellipse_center": ellipse_center,
        "ellipse_a1": max(axes_len),
        "ellipse_a2": min(axes_len),
        "ellipse_angle": adjusted_angle,
    }

    # plotting below
    # Draw the ellipse on a blank mask
    ellipse_mask = np.zeros_like(img)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    # Draw the circle on a blank mask
    circle_mask = np.zeros_like(img)
    cv2.circle(circle_mask, center, radius, 255, -1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_labeled = draw_mask_contour(
        img_rgb, ellipse_mask, color=(255, 0, 0), thickness=1
    )
    img_labeled = draw_mask_contour(
        img_rgb, circle_mask, color=(0, 0, 255), thickness=1
    )

    if plot_save_dir:
        Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
        # assert Path(plot_save_dir).is_dir(), "plot_save_dir is not a valid directory."
        save_path = Path(plot_save_dir) / Path(pupil_image).name
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(img_labeled)
        plt.title("Labeled")
        plt.axis("off")
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    if debug_plot_save_dir:
        Path(debug_plot_save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(debug_plot_save_dir) / (Path(pupil_image).stem + "_debug.png")
        fig = plt.figure(figsize=(3, 3 * 8))

        x, y = get_center_of_mass(img_enhanced)
        plt.subplot(8, 1, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original Grayscale")
        plt.axis("off")

        plt.subplot(8, 1, 2)
        plt.imshow(img_enhanced, cmap="hot")
        plt.scatter([x], [y], color="cyan", marker="+", s=80, label="Center of Mass")
        plt.title("Contrast Enhanced")
        plt.axis("off")

        x, y = get_center_of_mass(img_smoothed)
        plt.subplot(8, 1, 3)
        plt.imshow(img_smoothed, cmap="hot")
        plt.scatter([x], [y], color="cyan", marker="+", s=80, label="Center of Mass")
        plt.title("Smoothed")
        plt.axis("off")

        x, y = get_center_of_mass(img_eroded)  # Note: (y, x) order
        plt.subplot(8, 1, 4)
        plt.imshow(img_eroded, cmap="hot")
        plt.scatter([x], [y], color="cyan", marker="+", s=80, label="Center of Mass")
        plt.title("Eroded")
        plt.axis("off")

        com_x, com_y = estimated_pupil_center
        plt.subplot(8, 1, 5)
        plt.imshow(img_threshed / 255, cmap="hot")
        plt.scatter(
            [com_x], [com_y], color="cyan", marker="+", s=80, label="Center of Mass"
        )
        plt.title("Threshed")
        plt.axis("off")

        plt.subplot(8, 1, 6)
        plt.imshow(img_significant_contours)
        plt.title("Significant Contours")
        plt.axis("off")

        plt.subplot(8, 1, 7)
        plt.imshow(img_filtered_contours)
        plt.title("Filtered Contours")
        plt.axis("off")

        plt.subplot(8, 1, 8)
        plt.imshow(img_labeled)
        plt.title("Labeled")
        plt.axis("off")

        plt.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    return info


# %%
if __name__ == "__main__":
    DATA_PATH = "./data_cropped/"
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
    image_file = (
        "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_67609.png"
    )
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

    pupil_image = Path(DATA_PATH) / image_file

    info = find_pupil_info(
        pupil_image,
        plot_save_dir="./sample_results",
        debug_plot_save_dir="./sample_results",
    )
