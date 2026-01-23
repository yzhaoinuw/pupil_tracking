# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:19:02 2025

@author: yzhao
"""


import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import InterpolationMode


def resize_with_pad(
    img: Image.Image,
    target_size: int = 148,
    fill: int = 0,
    resample=Image.BILINEAR,
) -> Image.Image:
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


def random_zoom_translate_pil(
    img: Image.Image,
    mask: Image.Image,
    target_size: int = 148,
    scale_range=(0.85, 1.15),
    fill_img: int = 0,
    fill_mask: int = 0,
    p: float = 0.5,
):
    """
    With probability p:
      - applies slight zoom in/out + random translation while keeping output size == target_size.
    Otherwise returns (img, mask) unchanged.
    """
    if random.random() > p:
        return img, mask

    s = random.uniform(*scale_range)
    new_size = int(round(target_size * s))
    new_size = max(1, new_size)

    img_rs = img.resize((new_size, new_size), resample=Image.BILINEAR)
    mask_rs = mask.resize((new_size, new_size), resample=Image.NEAREST)

    if new_size == target_size:
        return img_rs, mask_rs

    if new_size > target_size:
        # zoom in: random crop
        max_left = new_size - target_size
        max_top = new_size - target_size
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)
        box = (left, top, left + target_size, top + target_size)
        return img_rs.crop(box), mask_rs.crop(box)

    # zoom out: random placement (translation + padding jitter)
    canvas_img = Image.new("L", (target_size, target_size), color=fill_img)
    canvas_mask = Image.new("L", (target_size, target_size), color=fill_mask)

    max_left = target_size - new_size
    max_top = target_size - new_size
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)

    canvas_img.paste(img_rs, (left, top))
    canvas_mask.paste(mask_rs, (left, top))
    return canvas_img, canvas_mask


def random_pad_and_crop_pil(
    img: Image.Image,
    mask: Image.Image,
    target_size: int = 148,
    max_pad: int = 12,
    fill_img: int = 0,
    fill_mask: int = 0,
    p: float = 0.5,
):
    """
    With probability p:
      - randomly pads each side by [0..max_pad], then random-crops back to target_size.
    Otherwise returns (img, mask) unchanged.
    """
    if random.random() > p:
        return img, mask

    pad_l = random.randint(0, max_pad)
    pad_r = random.randint(0, max_pad)
    pad_t = random.randint(0, max_pad)
    pad_b = random.randint(0, max_pad)

    w, h = img.size  # expected (target_size, target_size)
    new_w = w + pad_l + pad_r
    new_h = h + pad_t + pad_b

    padded_img = Image.new("L", (new_w, new_h), color=fill_img)
    padded_mask = Image.new("L", (new_w, new_h), color=fill_mask)
    padded_img.paste(img, (pad_l, pad_t))
    padded_mask.paste(mask, (pad_l, pad_t))

    max_left = new_w - target_size
    max_top = new_h - target_size
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    box = (left, top, left + target_size, top + target_size)

    return padded_img.crop(box), padded_mask.crop(box)


class PupilDataset(Dataset):
    def __init__(
        self,
        image_paths,
        mask_paths=None,
        augment=False,
        target_size=148,
        scale_range=(0.85, 1.15),
        max_pad=12,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.target_size = target_size
        self.scale_range = scale_range
        self.max_pad = max_pad

        self.pil_to_tensor = transforms.PILToTensor()

        # Geometric transforms for (img, mask) together
        self.transform_img_mask = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=45,
                    interpolation=InterpolationMode.NEAREST,  # keep mask crisp
                ),
            ]
        )

        # Photometric transforms for image only
        self.transform_img = transforms.Compose(
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")
        img = resize_with_pad(img, target_size=self.target_size, resample=Image.BILINEAR)

        if self.mask_paths is None:
            img = self.pil_to_tensor(img).float() / 255.0
            return img, img_path.name

        mask = Image.open(self.mask_paths[idx]).convert("L")
        mask = resize_with_pad(mask, target_size=self.target_size, resample=Image.NEAREST)

        if self.augment:
            # zoom/translate/pad jitter
            img, mask = random_zoom_translate_pil(
                img, mask,
                target_size=self.target_size,
                scale_range=self.scale_range,
                fill_img=0, 
                fill_mask=0,
                p=0.7
            )
            img, mask = random_pad_and_crop_pil(
                img, mask,
                target_size=self.target_size,
                max_pad=self.max_pad,
                fill_img=0, 
                fill_mask=0,
                p=0.7
            )

            # flips + rotation
            img, mask = self.transform_img_mask(img, mask)

            # image-only photometric
            img = self.transform_img(img)

        img = self.pil_to_tensor(img).float() / 255.0

        mask = self.pil_to_tensor(mask).float()
        mask = (mask > 0).float()  # ensure {0,1}

        return img, mask
