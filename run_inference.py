# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 00:34:33 2025

@author: yzhao
"""

import os
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image


from unet_sketch import UNet


class TestDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = list(Path(image_dir).glob("*.png"))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.CenterCrop((148, 148)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image, img_path.name


checkpoint_dir = Path("checkpoints")
checkpoint_path = checkpoint_dir / "best_model_iou=0.8837.pth"
test_dataset = TestDataset("images_test/")
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

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
            orig = Image.open(Path("images_test") / names[i]).convert("L")
            orig = transforms.CenterCrop((148, 148))(orig)
            orig_np = np.array(orig)

            # Create RGB image from grayscale
            rgb = np.stack([orig_np] * 3, axis=-1)

            # Overlay red where mask is 1
            mask = preds[i].squeeze()  # shape: (H, W)
            overlay = rgb.copy()
            overlay[mask == 1] = [255, 0, 0]  # red where mask is positive

            # Optional: blend with original for transparency
            alpha = 0.8
            blended = (alpha * rgb + (1 - alpha) * overlay).astype(np.uint8)

            # Save result
            out_path = f"{result_dir}/{names[i]}"
            Image.fromarray(blended).save(out_path)
