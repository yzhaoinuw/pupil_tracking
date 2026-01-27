# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 11:37:09 2025

@author: yzhao
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from PIL import Image

# -------------------------
# User settings
# -------------------------
csv_path = Path(
    "movies/250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57_result/250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_estimated_pupil_diameter.csv"
)
image_parent = Path(
    "./movies/250530_5003_Green_Training_very_dm_light_2025-05-30T09-27-57.042_mask"
)  # <-- change this to your overlay dir
out_gif = Path("pupil_diameter_analysis_result_demo.gif")
sample_every = 3  # sample every Nth frame to keep it shorter
fps = 5  # output GIF frame rate
n_frames = 90  # max frames to include (optional)
# -------------------------

# Read CSV
df = pd.read_csv(csv_path)
df = df[df["estimated_pupil_diameter"] > 15]  # skip tiny pupil cases

# Sort by image_name if needed
# df = df.sort_values("image_name").reset_index(drop=True)

# Sampling
df = df.iloc[7100::sample_every].reset_index(drop=True)
if n_frames:
    df = df.head(n_frames)

# Load images
images = [np.array(Image.open(image_parent / name)) for name in df["image_name"]]
diameters = df["estimated_pupil_diameter"].to_numpy()

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax_img, ax_plot = axes
plt.tight_layout(pad=2.0)

# Initialize
im_display = ax_img.imshow(images[0], cmap="gray")
ax_img.set_title("Segmented Pupil")
ax_img.axis("off")

(line,) = ax_plot.plot([], [], lw=2)
(dot,) = ax_plot.plot([], [], "ro")
ax_plot.set_xlim(0, len(diameters))
ax_plot.set_ylim(diameters.min() * 0.9, diameters.max() * 1.1)
ax_plot.set_xlabel("Frame")
ax_plot.set_ylabel("Estimated Diameter (pixels)")
ax_plot.set_title("Pupil Diameter Over Time")


# Update function
def update(i):
    im_display.set_array(images[i])
    line.set_data(np.arange(i + 1), diameters[: i + 1])
    # dot.set_data(i, diameters[i])
    dot.set_data([i], [diameters[i]])
    return [im_display, line, dot]


anim = FuncAnimation(fig, update, frames=len(images), interval=1000 / fps, blit=True)

# Save as GIF (requires imagemagick or pillow)
anim.save(out_gif, writer="pillow", fps=fps)
plt.close()

print(f"✅ Saved animated GIF to: {out_gif}")
