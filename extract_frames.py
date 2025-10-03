# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 15:27:03 2025

@author: yzhao
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def extract_selected_frames(video_path, out_dir, extraction_fps=5, max_frames=10000):
    video_path = Path(video_path)
    video_name = video_path.stem
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}. Original FPS: {fps:.2f}")

    duration = frame_count / fps
    extraction_fps = min(extraction_fps, fps)
    n_frames = round(duration * extraction_fps)
    n_frames = min(max_frames, n_frames)
    extraction_fps = n_frames / duration
    print(f"Extracting {n_frames} frames at ~{extraction_fps:.2f} fps")

    selected_frames = np.linspace(0, frame_count - 1, n_frames, dtype=int)

    for i, frame_idx in tqdm(
        enumerate(selected_frames),
        total=n_frames,
        desc="Extracting frames",
        unit="frame",
    ):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {frame_idx}")
            continue

        out_path = out_dir / f"{video_name}_{i:05d}.png"
        cv2.imwrite(str(out_path), frame)

    cap.release()
    print(f"Extracted {len(selected_frames)} frames into {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract evenly spaced frames from a video."
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        required=True,
        help="Path to input video file (e.g., movie.avi)",
    )
    parser.add_argument(
        "--out_dir", type=Path, required=True, help="Directory to save extracted frames"
    )
    parser.add_argument(
        "--extraction_fps",
        type=float,
        default=5,
        help="Target extraction FPS (default: 5)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=10000,
        help="Maximum number of frames to extract (default: 10000)",
    )

    args = parser.parse_args()
    extract_selected_frames(
        args.video_path, args.out_dir, args.extraction_fps, args.max_frames
    )


if __name__ == "__main__":
    main()
