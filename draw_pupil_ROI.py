# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 19:24:28 2025

@author: yzhao
"""

import cv2
import numpy as np


points = []


def draw_point(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")


def main(image_path, output_path):
    global points
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.namedWindow("Draw ROI", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Draw ROI", draw_point)

    print("Click to add points. Press 's' to save, 'r' to reset, 'q' to quit.")

    while True:
        display_img = color_img.copy()
        for i, pt in enumerate(points):
            cv2.circle(display_img, pt, 3, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display_img, points[i - 1], pt, (0, 255, 0), 1)
        if len(points) > 2:
            # Draw closing line
            cv2.line(display_img, points[-1], points[0], (0, 255, 0), 1)

        cv2.imshow("Draw ROI", display_img)
        key = cv2.waitKey(1)

        if key == ord("s"):
            if len(points) >= 3:
                ROI = np.array(points, dtype=np.int32)
                np.save(output_path, ROI)
                print(f"Saved ROI to {output_path}")
                break
            else:
                print("At least 3 points required.")
        elif key == ord("r"):
            points = []
            print("Reset points.")
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    image_path = "average_pupil_ROI.png"
    main(image_path, "pupil_ROI.npy")
