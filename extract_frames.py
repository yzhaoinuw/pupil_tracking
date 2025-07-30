# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:53:22 2025

@author: yzhao
"""

import cv2
from pathlib import Path


VIDEO_PATH = Path("./movies/")
OUTPUT_DIR = Path("./data")  # where to save extracted frames
OUTPUT_DIR.mkdir(exist_ok=True)

movie_name = "250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042"
movie_file = VIDEO_PATH / (movie_name + ".avi")
cap = cv2.VideoCapture(movie_file)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Example: extract every 5th frame (change step if needed)
frame_idx = 0
saved_idx = 0
max_frames = 1000
step = max(30, frame_count // max_frames)  # or 1 to extract all
print(f"Total frames: {frame_count}, FPS: {fps}, Step: {step}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % step == 0:
        fname = OUTPUT_DIR / f"{movie_name}_{frame_idx:04d}.png"
        cv2.imwrite(str(fname), frame)
        saved_idx += 1

    frame_idx += 1

cap.release()
