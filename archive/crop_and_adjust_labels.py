from pathlib import Path

import cv2
import pandas as pd

# --- USER SETTINGS ---
LABELS_CSV = "labels.csv"
IMAGES_DIR = "./data"
OUTPUT_LABELS = "labels_cropped.csv"
CROP_CENTER = (152, 88)
CROP_SIZE = 150
PREVIEW_INDEX = 0  # Row index in CSV to preview cropped image
# ----------------------


def crop_and_adjust_labels(df, crop_center, crop_size):
    x_offset, y_offset = (
        crop_center[0] - crop_size // 2,
        crop_center[1] - crop_size // 2,
    )
    df_new = df.copy()

    for i in range(1, 11):  # Assuming up to x10/y10
        x_col, y_col = f"x{i}", f"y{i}"
        if x_col in df.columns and y_col in df.columns:
            df_new[x_col] = df[x_col] - x_offset
            df_new[y_col] = df[y_col] - y_offset

    return df_new


def preview_crop(image_path, crop_center, crop_size):
    img = cv2.imread(str(image_path))
    x, y = crop_center
    half = crop_size // 2
    cropped = img[y - half : y + half, x - half : x + half]
    cv2.imshow("Cropped Preview", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop_all_images(input_dir, output_dir, crop_center, crop_size):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    cx, cy = crop_center
    half = crop_size // 2
    x_start, x_end = cx - half, cx + half
    y_start, y_end = cy - half, cy + half

    failed = []

    for img_path in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None or img.shape[0] < y_end or img.shape[1] < x_end:
            failed.append(img_path.name)
            continue
        cropped = img[y_start:y_end, x_start:x_end]
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), cropped)

    print(f"✅ Cropped {len(list(output_dir.glob('*.png')))} images.")
    if failed:
        print("⚠️ These images failed to process:", failed)


def main():
    df = pd.read_csv(LABELS_CSV)
    preview_frame = df.iloc[PREVIEW_INDEX]["frame"]
    # preview_path = Path(IMAGES_DIR) / preview_frame
    preview_path = Path(IMAGES_DIR) / preview_frame

    print(f"Previewing: {preview_frame}")
    preview_crop(preview_path, CROP_CENTER, CROP_SIZE)

    df_new = crop_and_adjust_labels(df, CROP_CENTER, CROP_SIZE)
    df_new.to_csv(OUTPUT_LABELS, index=False)
    print(f"✅ Cropped labels saved to {OUTPUT_LABELS}")


if __name__ == "__main__":
    # main()
    crop_all_images(IMAGES_DIR, "./data_cropped", crop_center=CROP_CENTER, crop_size=CROP_SIZE)
