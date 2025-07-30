import cv2
from pathlib import Path


def crop_centered_150x150(input_dir, output_dir):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(exist_ok=True)

    failed = []

    for img_path in sorted(input_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Failed to read: {img_path.name}")
            failed.append(img_path.name)
            continue

        h, w = img.shape[:2]
        if h < 150 or w < 150:
            print(f"⚠️ Too small to crop: {img_path.name}")
            failed.append(img_path.name)
            continue

        cx, cy = w // 2, h // 2
        x_start, x_end = cx - 75, cx + 75
        y_start, y_end = cy - 75, cy + 75

        cropped = img[y_start:y_end, x_start:x_end]
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), cropped)

    print(f"✅ Cropped {len(list(output_dir.glob('*.png')))} images.")
    if failed:
        print("⚠️ Failed to process:", failed)


if __name__ == "__main__":
    crop_centered_150x150("./data", "data_cropped")
