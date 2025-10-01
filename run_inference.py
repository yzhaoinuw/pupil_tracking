# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 00:34:33 2025

@author: yzhao
"""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image

from unet import UNet
from dataset_sketch import PupilDataset


checkpoint_dir = Path("checkpoints")
checkpoint_path = checkpoint_dir / "best_model_iou=0.8837.pth"
image_dir = "images_test/"

# Optional: blend with original for transparency
alpha = 0.9

image_paths = sorted(Path(image_dir).glob("*.png"))
test_dataset = PupilDataset(image_paths)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = UNet(use_attention=True)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
model.to("cuda" if torch.cuda.is_available() else "cpu")

result_dir = "predictions_test"
os.makedirs(result_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for images, names in test_loader:
        images = images.to(device)
        preds = model(images)
        preds = (preds > 0.6).float().cpu().numpy()

        for i in range(len(images)):
            # Load the original image from disk again (for visualization)
            orig = Image.open(Path(image_dir) / names[i]).convert("L")
            orig = transforms.CenterCrop((148, 148))(orig)
            orig_np = np.array(orig)

            # Create RGB image from grayscale
            rgb = np.stack([orig_np] * 3, axis=-1)

            # Overlay red where mask is 1
            mask = preds[i].squeeze()  # shape: (H, W)
            overlay = rgb.copy()
            overlay[mask == 1] = [255, 0, 0]  # red where mask is positive

            blended = (alpha * rgb + (1 - alpha) * overlay).astype(np.uint8)

            # Save result
            out_path = f"{result_dir}/{names[i]}"
            Image.fromarray(blended).save(out_path)
