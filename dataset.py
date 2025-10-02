# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:19:02 2025

@author: yzhao
"""

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms


class PupilDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, augment=False, crop_size=148):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.crop_size = crop_size
        self.pil_to_tensor = transforms.PILToTensor()
        self.center_crop = transforms.CenterCrop(self.crop_size)

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
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")
        img = self.center_crop(img)

        if self.mask_paths is None:
            img = self.pil_to_tensor(img)
            img = img.float()
            img = img / 255.0
            return img, img_path.name

        mask = Image.open(self.mask_paths[idx])
        mask = self.center_crop(mask)
        if self.augment:
            img, mask = self.transform_img_mask(img, mask)
            img = self.transform_img(img)

        img, mask = self.pil_to_tensor(img), self.pil_to_tensor(mask)
        img, mask = img.float(), mask.float()
        img = img / 255.0
        return img, mask
