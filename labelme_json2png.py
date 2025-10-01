# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 20:25:21 2025

@author: yzhao
"""

from pathlib import Path
import subprocess
import shutil

image_dir = Path("./images_train")
mask_dir = Path("./masks_train")
mask_dir.mkdir(exist_ok=True, parents=True)

for json_file in image_dir.glob("*.json"):

    subfolder_name = json_file.stem
    existing_label_path = mask_dir / f"{subfolder_name}.png"
    if existing_label_path.exists():
        continue

    print(f"Processing {json_file.name}")
    subprocess.run(["labelme_export_json", str(json_file)])
    subfolder_dir = image_dir / subfolder_name
    label_path = subfolder_dir / "label.png"
    if label_path.exists():
        print(label_path.name)
        shutil.move(str(label_path), mask_dir / f"{subfolder_name}.png")
        shutil.rmtree(subfolder_dir)
