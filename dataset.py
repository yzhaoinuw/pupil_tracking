# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:19:02 2025

@author: yzhao
"""

import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms.v2 as transforms


pil_to_tensor = transforms.PILToTensor()
crop = transforms.CenterCrop(148)


def center_crop(img: torch.Tensor, size: int = 148) -> torch.Tensor:
    _, h, w = img.shape  # [C,H,W]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[:, top : top + size, left : left + size]


class PupilTrainDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def transform_img_mask(self, img, mask):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
            ]
        )
        return transform(img, mask)

    def transform_img(self, img):
        transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.2,  # brightness factor: 0.8 to 1.2
                            contrast=0.2,  # contrast factor: 0.8 to 1.2
                        )
                    ],
                    p=0.5,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5
                ),
            ]
        )
        return transform(img)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  # flatten channel
        mask = Image.open(self.mask_paths[idx])
        img, mask = crop(img), crop(mask)

        if self.augment:
            img, mask = self.transform_img_mask(img, mask)
            img = self.transform_img(img)

        img = pil_to_tensor(img)
        mask = pil_to_tensor(mask)
        img, mask = img.float(), mask.float()
        img = img / 255.0
        return img, mask


class PupilTestDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(self.image_paths[idx]).convert("L")
        img = crop(img)
        img = pil_to_tensor(img)
        img = img.float()
        img = img / 255.0
        return img, img_path.name
