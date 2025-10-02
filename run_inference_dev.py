# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 00:34:33 2025

@author: yzhao
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from unet import UNet
from dataset import PupilDataset


# %% input args
image_dir = Path("images_test_2")
result_dir = Path("results")

# optional
checkpoint_path = (
    Path("checkpoints") / "best_model_iou=0.8837.pth"
)  # default to this if not provided

output_img_dir = Path("predicted_masks_2")
# output_img_dir = None
# %%
result_dir.mkdir(parents=True, exist_ok=True)

if output_img_dir is not None:
    output_img_dir.mkdir(parents=True, exist_ok=True)

alpha = 0.9
image_paths = sorted(image_dir.glob("*.png"))
test_dataset = PupilDataset(image_paths)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = UNet(use_attention=True)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

pupil_diams = []
with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)
        preds = model(images)
        preds = (preds > 0.6).float().cpu().numpy()
        pupil_diam_batch = np.sqrt(np.sum(preds, axis=(1, 2, 3)))

        for i, (name, diam) in enumerate(zip(names, pupil_diam_batch)):
            pupil_diams.append((name, diam))

            if output_img_dir is not None:
                # Load the original image from disk again (for visualization)
                orig = Image.open(image_dir / names[i]).convert("L")
                orig = test_dataset.center_crop(orig)
                orig_np = np.array(orig)

                # Create RGB image from grayscale
                rgb = np.stack([orig_np] * 3, axis=-1)

                # Overlay red where mask is 1
                mask = preds[i].squeeze()  # shape: (H, W)
                overlay = rgb.copy()
                overlay[mask == 1] = [255, 0, 0]  # red where mask is positive

                blended = (alpha * rgb + (1 - alpha) * overlay).astype(np.uint8)

                # Save result
                out_path = output_img_dir / names[i]
                Image.fromarray(blended).save(out_path)

# %%
exp_name = "_".join(pupil_diams[0][0].split("_")[:-1])
pupil_diams.sort(key=lambda x: int(Path(x[0]).stem.split("_")[-1]))
df = pd.DataFrame(pupil_diams, columns=["image_name", "estimated_pupil_diameter"])
df.index = np.arange(1, len(df) + 1)
df.to_csv(result_dir / f"{exp_name}_estimated_pupil_diameter.csv", index=True)

plt.figure(figsize=(10, 6))
plt.plot(df["estimated_pupil_diameter"], linewidth=1)
plt.xlabel("Frame")
plt.ylabel("Estimated Pupil Diameter (pixels)")
plt.title("Estimated Pupil Diameter Over Time")
plt.tight_layout()
plt.savefig(result_dir / f"{exp_name}_estimated_pupil_diameter.png", dpi=200)
plt.show()
