# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 14:28:50 2025

@author: yzhao
"""

from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd

# Constants
MAX_POINTS = 10
FRAME_FOLDER = Path("./data_cropped_centered")
LABELS_CSV = Path("labels.csv")


class Labeler:
    def __init__(self):
        self.points = []

    def reset_points(self):
        self.points = []

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < MAX_POINTS:
            self.points.append((x, y))


def extract_recording_and_index(filename):
    stem = filename.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        recording, index = parts
        return recording, int(parts[1])
    else:
        return stem, -1  # fallback


def get_sorted_image_paths(frame_folder):
    frames_by_recording = defaultdict(list)
    for path in frame_folder.glob("*.png"):
        recording, index = extract_recording_and_index(path)
        frames_by_recording[recording].append((index, path))

    sorted_paths = []
    for recording in sorted(frames_by_recording.keys()):
        sorted_frames = sorted(frames_by_recording[recording], key=lambda x: x[0])
        sorted_paths.extend([path for _, path in sorted_frames])

    return sorted_paths


def label_frame(image, frame_name, labeler, window_name="Label Frame"):
    labeler.reset_points()
    clone = image.copy()
    cv2.setMouseCallback(window_name, labeler.click_event)

    while True:
        display = clone.copy()
        for idx, (x, y) in enumerate(labeler.points):
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"{idx+1}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        video_name, frame_ind = frame_name.rsplit("_", 1)
        if len(video_name) > 7:
            video_name = video_name[:10] + "..."
        frame_name_truncated = "_".join([video_name, frame_ind])
        cv2.putText(
            display,
            frame_name_truncated,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            action = "next"
            break
        elif key == ord("j"):
            action = "jump"
            break
        elif key == ord("r"):
            labeler.reset_points()
        elif key == ord("q"):
            action = "quit"
            break

    if not labeler.points:
        return None, action

    label_row = {"frame": frame_name}
    for i in range(MAX_POINTS):
        if i < len(labeler.points):
            label_row[f"x{i+1}"] = labeler.points[i][0]
            label_row[f"y{i+1}"] = labeler.points[i][1]
        else:
            label_row[f"x{i+1}"] = None
            label_row[f"y{i+1}"] = None

    return label_row, action


def main():
    image_paths = get_sorted_image_paths(FRAME_FOLDER)
    if not image_paths:
        print(f"No PNG images found in {FRAME_FOLDER.resolve()}")
        return

    all_labels = []
    if LABELS_CSV.exists():
        all_labels = pd.read_csv(LABELS_CSV).to_dict("records")

    labeled_frames = set(label["frame"] for label in all_labels)
    labeler = Labeler()
    WINDOW_NAME = "Label Frame"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    i = 0
    while i < len(image_paths):
        img_path = image_paths[i]
        frame_name = img_path.name
        if frame_name in labeled_frames:
            i += 1
            continue

        image = cv2.imread(str(img_path))
        label_data, action = label_frame(image, frame_name, labeler, window_name=WINDOW_NAME)

        if label_data is not None:
            all_labels.append(label_data)
            pd.DataFrame(all_labels).to_csv(LABELS_CSV, index=False)
            print(f"Labeled {frame_name}")
        else:
            print(f"Skipped {frame_name} (no points)")

        if action == "quit":
            print("User exited.")
            break
        elif action == "next":
            i += 1
            continue
        elif action == "jump":
            i += 100
            continue

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
