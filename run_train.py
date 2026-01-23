# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 23:09:41 2025

@author: yzhao
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unet import UNet
from dataset import PupilDataset


# -------------------- Metrics -------------------- #
def dice_score(pred, target, pred_thresh=0.6, epsilon=1e-6):
    pred = (pred > pred_thresh).float()
    target = (target > pred_thresh).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + epsilon)


def iou_score(pred, target, pred_thresh=0.6, epsilon=1e-6):
    pred = (pred > pred_thresh).float()
    target = (target > pred_thresh).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + epsilon)


# -------------------- Data Preparation -------------------- #
def make_dataset(image_dir, mask_dir, augment=False):
    image_paths = sorted(Path(image_dir).glob("*.png"))
    mask_paths = sorted(Path(mask_dir).glob("*.png"))
    return PupilDataset(image_paths, mask_paths, augment=augment)


# %% Hyperparameters setup
pred_thresh = 0.7
notable_iou = 0.85
patience = 5
n_epochs = 200
checkpoint_dir = Path("checkpoints")

train_dataset = make_dataset("images_train", "masks_train", augment=True)
val_dataset = make_dataset("images_validation", "masks_validation", augment=False)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

model_name = f"unet_atn_resize_{len(train_dataset)}pupils_thresh={pred_thresh}"

# %% -------------------- Model Setup -------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(use_attention=True).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",  # Because we are maximizing IoU
    factor=0.5,  # Reduce LR by half
    patience=10,  # Wait 3 epochs with no improvement
    min_lr=1e-3 * 0.5**5,  # Set a floor to avoid vanishing LR
)

# %% -------------------- Training Loop -------------------- #
best_val_loss = float("inf")
best_val_iou = 0
prev_iou = 0
patience_counter = 0
log_lines = []  # ← log storage

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------------------- Validation -------------------- #
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            val_loss += loss.item()

            for j in range(imgs.shape[0]):
                dice = dice_score(preds[j], masks[j], pred_thresh=pred_thresh)
                iou = iou_score(preds[j], masks[j], pred_thresh=pred_thresh)
                val_dice += dice
                val_iou += iou

    val_loss /= len(val_loader)
    val_dice /= len(val_dataset)
    val_iou /= len(val_dataset)

    # -------------------- Logging -------------------- #
    log_line = (
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_dice:.4f} | "
        f"Val IoU: {val_iou:.4f} | "
        f"Learning Rate: {scheduler.get_last_lr()[0]}"
    )
    print(log_line)
    log_lines.append(log_line)
    scheduler.step(val_iou)
    # -------------------- Early Stopping -------------------- #
    if val_iou > prev_iou:
        if val_iou > best_val_iou:
            best_val_iou = val_iou  # ← save for filename later
            if best_val_iou > notable_iou:
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / f"{model_name}_iou={best_val_iou:.4f}.pth",
                )
                print("✅ New best model saved!")

        patience_counter = 0
    else:
        patience_counter += 1
        print(f"⏳ Patience: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("⛔ Early stopping triggered.")
            break

    prev_iou = val_iou

if best_val_iou > notable_iou:
    # -------------------- Save Log File -------------------- #
    log_filename = f"training_log_{model_name}_iou={best_val_iou:.4f}.txt"
    with open(checkpoint_dir / log_filename, "w") as f:
        f.write("\n".join(log_lines))
    print(f"📝 Training log saved as: {log_filename}")
