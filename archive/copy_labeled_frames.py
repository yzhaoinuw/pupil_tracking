# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 17:01:45 2025

@author: yzhao
"""

import shutil

import pandas as pd
from pathlib import Path


# Paths (adjust if needed)
labels_xlsx = Path("C:/Users/yzhao/python_projects/pupil_tracking/labels.csv")
source_dir = Path("C:/Users/yzhao/python_projects/pupil_tracking/data")
dest_dir = Path(
    "C:/Users/yzhao/python_projects/pupil_tracking/PupilTracking-YueZhao-2025-07-23/labeled-data/250616_5120_Purple_sleep_trial_1_2025-06-16T16-31-19.701"
)

# Create destination dir if not exists
dest_dir.mkdir(parents=True, exist_ok=True)

# Load labeled frame names
df = pd.read_csv(labels_xlsx)
frame_names = df["frame"].tolist()

# Copy files
missing = []
for frame_name in frame_names:
    src_path = source_dir / frame_name
    dst_path = dest_dir / frame_name
    if src_path.exists():
        shutil.copy(src_path, dst_path)
    else:
        missing.append(frame_name)

print(f"✅ Copied {len(frame_names) - len(missing)} frames.")
if missing:
    print(f"⚠️ Missing {len(missing)} frames:")
    for name in missing:
        print("  -", name)
else:
    print("🎉 All labeled frames copied successfully!")
