# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 00:07:48 2025

@author: yzhao
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from unet import UNet
from dataset import PupilDataset
from extract_frames import extract_selected_frames  # <--- NEW IMPORT


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "best_model_iou=0.8837.pth"


def generate_pupil_mask_prediction(
    checkpoint_path,
    image_dir: Path,
    output_mask_dir: Path = None,
    pred_thresh: float = 0.6,
    batch_size: int = 32,
    mask_transparency: float = 0.1,
):
    """Run inference, collect pupil diameters, and optionally save overlay images."""
    print(f"Building dataloader with batch size = {batch_size}.")
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG files found in {image_dir}")

    test_dataset = PupilDataset(image_paths)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    print("Loading UNet model...")
    model = UNet(use_attention=True)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    results = []
    with torch.inference_mode():
        for images, names in tqdm(
            test_loader, desc="Segmenting pupil images...", unit="batch"
        ):
            images = images.to(device)
            with torch.autocast(device_type=device_name, dtype=torch.float16):
                preds = model(images)
            preds = (preds > pred_thresh).float().cpu().numpy()
            pupil_diam_batch = np.sqrt(np.sum(preds, axis=(1, 2, 3)))

            for i, (name, diam) in enumerate(zip(names, pupil_diam_batch)):
                results.append((name, diam))

                if output_mask_dir is not None:
                    orig = Image.open(image_dir / names[i]).convert("L")
                    orig = test_dataset.center_crop(orig)
                    orig_np = np.array(orig)
                    rgb = np.stack([orig_np] * 3, axis=-1)
                    mask = preds[i].squeeze()
                    overlay = rgb.copy()
                    overlay[mask == 1] = [255, 0, 0]
                    blended = (
                        (1 - mask_transparency) * rgb + mask_transparency * overlay
                    ).astype(np.uint8)
                    out_path = output_mask_dir / names[i]
                    Image.fromarray(blended).save(out_path)

    return results


def save_results(results, result_dir: Path, exp_name: str):
    results.sort(key=lambda x: int(Path(x[0]).stem.split("_")[-1]))
    df = pd.DataFrame(results, columns=["image_name", "estimated_pupil_diameter"])
    df.index = np.arange(1, len(df) + 1)
    csv_path = result_dir / f"{exp_name}_estimated_pupil_diameter.csv"
    df.to_csv(csv_path, index=True)

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["estimated_pupil_diameter"], linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Estimated Pupil Diameter (pixels)")
    plt.title("Estimated Pupil Diameter Over Time")
    plt.tight_layout()
    plot_path = result_dir / f"{exp_name}_estimated_pupil_diameter.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"Saved CSV:  {csv_path}")
    print(f"Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Pupil diameter analysis pipeline")
    parser.add_argument(
        "--video_path", type=Path, help="Optional video file to extract frames from"
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        help="Directory to save extracted frames (used with --video_path)",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        help="Directory of existing PNG images (skips extraction)",
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        help="Directory to save results (auto-created if not given)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=Path,
        default=None,
        help="Optional directory to save overlay images",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pred_thresh", type=float, default=0.6)
    parser.add_argument("--mask_transparency", type=float, default=0.1)
    parser.add_argument("--extraction_fps", type=float, default=5)
    parser.add_argument("--max_frames", type=int, default=10000)
    args = parser.parse_args()

    # Handle extraction logic
    if args.video_path:
        print("Video path provided — extracting frames first...")

        # Auto-derive out_dir if not given
        if args.out_dir is None:
            args.out_dir = args.video_path.parent / f"{args.video_path.stem}_frames"
            print(f"No out_dir provided. Using default: {args.out_dir}")

        extract_selected_frames(
            args.video_path, args.out_dir, args.extraction_fps, args.max_frames
        )
        args.image_dir = args.out_dir

    elif args.image_dir is None:
        raise ValueError("You must specify either (--video_path) or (--image_dir).")

    if args.output_mask_dir is not None:
        args.output_mask_dir.mkdir(parents=True, exist_ok=True)

    results = generate_pupil_mask_prediction(
        args.checkpoint,
        args.image_dir,
        output_mask_dir=args.output_mask_dir,
        pred_thresh=args.pred_thresh,
        batch_size=args.batch_size,
        mask_transparency=args.mask_transparency,
    )

    if args.result_dir is None:
        args.result_dir = args.image_dir.parent / f"{args.image_dir.stem}_result"
        print(f"No result_dir provided. Using default: {args.result_dir}")
    args.result_dir.mkdir(parents=True, exist_ok=True)
    exp_name = (
        "_".join(Path(results[0][0]).stem.split("_")[:-1]) if results else "experiment"
    )
    save_results(results, args.result_dir, exp_name)


if __name__ == "__main__":
    main()
