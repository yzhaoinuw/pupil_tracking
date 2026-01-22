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

    def __len__(self):
        return len(self.image_paths)
    
    def resize_with_pad(self, img: Image.Image, target_size: int = 148, fill: int = 0, resample=Image.BILINEAR):
        w, h = img.size
        if w >= h:
            new_w = target_size
            new_h = int(round(h * target_size / w))
        else:
            new_h = target_size
            new_w = int(round(w * target_size / h))

        img = img.resize((new_w, new_h), resample=resample)

        pad_w = target_size - new_w
        pad_h = target_size - new_h
        left = pad_w // 2
        top = pad_h // 2

        padded = Image.new("L", (target_size, target_size), color=fill)
        padded.paste(img, (left, top))
        return padded

    def transform_img_mask(self, img: Image.Image, mask: Image.Image):
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45, interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )
        return transform(img, mask)

    def transform_img(self, img: Image.Image):
        transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2)],
                    p=0.5,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                    p=0.5,
                ),
            ]
        )
        return transform(img)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")
        img = self.resize_with_pad(img, target_size=148, resample=Image.BILINEAR)

        if self.mask_paths is None:
            img = self.pil_to_tensor(img).float() / 255.0
            return img, img_path.name

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = self.resize_with_pad(mask, target_size=148, resample=Image.NEAREST)

        if self.augment:
            img, mask = self.transform_img_mask(img, mask)
            img = self.transform_img(img)

        img = self.pil_to_tensor(img).float() / 255.0
        mask = self.pil_to_tensor(mask).float()
        mask = (mask > 0).float()
        return img, mask

