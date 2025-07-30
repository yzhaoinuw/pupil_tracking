# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:22:00 2025

@author: yzhao
"""

import pandas as pd
from pathlib import Path


# === SETTINGS ===
labels_csv = Path("C:/Users/yzhao/python_projects/pupil_tracking/labels.csv")
project_path = Path(
    r"C:\Users\yzhao\python_projects\pupil_tracking\PupilTracking-YueZhao-2025-07-23"
)
video_name = "250616_5120_Purple_sleep_trial 1_2025-06-16T16-31-19.701"
scorer = "YueZhao"
bodyparts = ["left_pupil_edge", "right_pupil_edge"]
config_path = project_path / "config.yaml"

# === OUTPUT PATH ===
csv_path = project_path / "labeled-data" / video_name / f"CollectedData_{scorer}.csv"
frame_dir = Path("labeled-data") / video_name

# === 1. Read labels from Excel
df = pd.read_csv(labels_csv)

# === 2. Construct image paths (relative)
image_names = df["frame"].apply(lambda f: str(frame_dir / f))

# === 3. Build correct MultiIndex columns
columns = pd.MultiIndex.from_tuples(
    [(scorer, bp, coord) for bp in bodyparts for coord in ["x", "y"]],
    names=["scorer", "bodyparts", "coords"],
)

dlc_df = pd.DataFrame(columns=columns, index=image_names)
dlc_df.index.name = None  # Important: no 'frame' header

# Fill in coordinates
dlc_df[(scorer, bodyparts[0], "x")] = df["x1"].values
dlc_df[(scorer, bodyparts[0], "y")] = df["y1"].values
dlc_df[(scorer, bodyparts[1], "x")] = df["x2"].values
dlc_df[(scorer, bodyparts[1], "y")] = df["y2"].values

# === 4. Write CSV
csv_path.parent.mkdir(parents=True, exist_ok=True)
dlc_df.to_csv(csv_path)
print(f"✅ DLC-compatible CSV written to: {csv_path}")
