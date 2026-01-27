# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 18:39:19 2025

@author: yzhao
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def average_images(folder_path, save_path=""):
    folder = Path(folder_path)
    image_files = sorted(
        [f for f in folder.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]]
    )

    if not image_files:
        print("No image files found.")
        sys.exit(1)

    # Load first image to get shape
    first_image = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        print(f"Failed to load image: {image_files[0]}")
        sys.exit(1)

    image_shape = first_image.shape
    acc = np.zeros(image_shape, dtype=np.float64)
    count = 0

    for img_path in image_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            continue
        if img.shape != image_shape:
            print(f"Skipping image with different shape: {img_path}")
            continue
        acc += img
        count += 1

    if count == 0:
        print("No valid images to average.")
        sys.exit(1)

    avg_img = (acc / count).astype(np.uint8)

    # Plot result
    plt.figure(figsize=(6, 6))
    plt.imshow(avg_img, cmap="gray")
    plt.title(f"Averaged Image ({count} images)")
    plt.axis("off")
    plt.show()
    if save_path:
        cv2.imwrite(save_path, avg_img)


# Example usage
# Replace './images' with your folder path
average_images("./pupil_range", "./average_pupil_ROI.png")
