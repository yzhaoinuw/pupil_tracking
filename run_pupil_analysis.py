# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 16:40:00 2025

@author: yzhao
"""


from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from pupil_segmentation import estimate_pupil_info


def process_one(pupil_file, plot_save_dir):
    result = estimate_pupil_info(pupil_file, plot_save_dir=plot_save_dir)
    result["image_file"] = pupil_file.name
    return result


if __name__ == "__main__":
    IMAGE_PATH = Path("./data_cropped_centered")
    pupil_files = sorted(IMAGE_PATH.glob("*.png"))
    plot_save_dir = Path("./data_segmented")
    plot_save_dir.mkdir(parents=True, exist_ok=True)

    csv_save_path = Path("./") / f"{IMAGE_PATH.name}_segmented.csv"

    # 🧵 Use 4 processes with tqdm progress bar
    rows = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        # futures = {executor.submit(process_one, f): f for f in pupil_files}
        futures = {
            executor.submit(process_one, pupil_file, plot_save_dir): pupil_file
            for pupil_file in pupil_files
        }
        for future in tqdm(
            as_completed(futures), total=len(pupil_files), desc="Processing"
        ):
            rows.append(future.result())

    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_save_path, index=False)
    print(f"Saved {len(df)} rows to: {csv_save_path}")
