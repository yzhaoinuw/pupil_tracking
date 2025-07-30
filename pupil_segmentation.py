# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 13:42:59 2025

@author: yzhao
"""

from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import remove_appendages, enhance_contrast, draw_mask_contour


def estimate_pupil_info(pupil_image, plot_save_dir="", debug_plot_save_dir=""):
    img = cv2.imread(pupil_image, cv2.IMREAD_GRAYSCALE)

    # contrast enhancement
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_enhanced = enhance_contrast(img)

    # Threshold and clean
    _, img_threshed = cv2.threshold(img_enhanced, 50, 255, cv2.THRESH_BINARY_INV)
    img_morphed = cv2.morphologyEx(
        img_threshed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_morphed = cv2.morphologyEx(img_morphed, cv2.MORPH_CLOSE, kernel)

    # appendage removal
    img_trimmed = remove_appendages(img_morphed)

    # Find contours
    contours, _ = cv2.findContours(
        img_trimmed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 15:  # arbitrary small blob threshold
        return {
            "center_circle": None,
            "diameter": None,
            "ellipse_center": None,
            "ellipse_a1": None,
            "ellipse_a2": None,
            "ellipse_angle": None,
        }

    largest = cv2.convexHull(largest)

    # Fit ellipse to the largest contour
    ellipse = cv2.fitEllipse(largest)

    # Fit a circle to the largest contour
    (center_x, center_y), radius = cv2.minEnclosingCircle(largest)
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
    overlay = img_rgb.copy()

    # Draw the ellipse on a blank mask
    ellipse_mask = np.zeros_like(img)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    # Draw the circle on a blank mask
    circle_mask = np.zeros_like(img)
    cv2.circle(circle_mask, center, radius, 255, -1)

    blended = draw_mask_contour(overlay, ellipse_mask, color=(255, 0, 0), thickness=1)
    blended = draw_mask_contour(overlay, circle_mask, color=(0, 0, 255), thickness=1)

    if plot_save_dir:
        Path(plot_save_dir).mkdir(parents=True, exist_ok=True)
        # assert Path(plot_save_dir).is_dir(), "plot_save_dir is not a valid directory."
        save_path = Path(plot_save_dir) / (Path(pupil_image).stem + "_segmented.png")
        fig = plt.figure(figsize=(3, 3))
        plt.imshow(blended)
        plt.title("Labeled")
        plt.axis("off")
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    if debug_plot_save_dir:
        Path(debug_plot_save_dir).mkdir(parents=True, exist_ok=True)
        # assert Path(debug_plot_save_dir).is_dir(), "debug_plot_save_dir is not a valid directory."
        save_path = Path(debug_plot_save_dir) / (Path(pupil_image).stem + "_debug.png")
        # Plot side-by-side
        fig = plt.figure(figsize=(3, 15))
        plt.subplot(5, 1, 1)
        plt.imshow(img, cmap="gray")
        plt.title("Original")
        plt.axis("off")

        plt.subplot(5, 1, 2)
        plt.imshow(img_enhanced, cmap="gray")
        plt.title("Contrast Enhanced")
        plt.axis("off")

        plt.subplot(5, 1, 3)
        plt.imshow(img_morphed, cmap="gray")
        plt.title("Smoothed and Morphed")
        plt.axis("off")

        plt.subplot(5, 1, 4)
        plt.imshow(img_trimmed, cmap="gray")
        plt.title("Appendage Removed")
        plt.axis("off")

        plt.subplot(5, 1, 5)
        plt.imshow(blended)
        plt.title("Labeled")
        plt.axis("off")

        plt.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        plt.show()

    return info


# %%
if __name__ == "__main__":

    DATA_PATH = "./data_cropped_centered/"
    # Load grayscale image
    # can't find contour
    image_file = (
        "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_3686.png"
    )

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
    pupil_image = Path(DATA_PATH) / image_file
    estimate_pupil_info(
        pupil_image,
        plot_save_dir="./sample_results",
        # debug_plot_save_dir=""
    )
