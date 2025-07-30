import cv2
import pandas as pd
from pathlib import Path
from collections import defaultdict

# --- Config ---
FRAME_FOLDER = Path("./data_cropped_centered")
SCORER_NAME = "YueZhao"
OUTPUT_CSV = Path(f"CollectedData_{SCORER_NAME}.csv")

MAX_POINTS = 10
BODY_PARTS = [f"pt{i+1}" for i in range(MAX_POINTS)]
# --------------


class Labeler:
    def __init__(self):
        self.points = []

    def reset_points(self):
        self.points = []

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < MAX_POINTS:
            self.points.append((x, y))


def extract_recording_and_index(filename):
    stem = filename.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        recording, index = parts
        return recording, int(parts[1])
    else:
        return stem, -1


def get_sorted_image_paths(frame_folder):
    frames_by_recording = defaultdict(list)
    for path in frame_folder.glob("*.png"):
        recording, index = extract_recording_and_index(path)
        frames_by_recording[recording].append((index, path))
    sorted_paths = []
    for recording in sorted(frames_by_recording.keys()):
        sorted_frames = sorted(frames_by_recording[recording], key=lambda x: x[0])
        sorted_paths.extend([path for _, path in sorted_frames])
    return sorted_paths


def label_frame(image, frame_name, labeler, window_name="Label Frame"):
    labeler.reset_points()
    clone = image.copy()
    cv2.setMouseCallback(window_name, labeler.click_event)

    while True:
        display = clone.copy()
        for idx, (x, y) in enumerate(labeler.points):
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                display,
                f"{idx+1}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        # Frame name overlay
        cv2.putText(
            display,
            frame_name,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return "closed"

        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            break
        elif key == ord("r"):
            labeler.reset_points()
        elif key == ord("q"):
            return "quit"

    if not labeler.points:
        return None

    row = [frame_name]
    for i in range(MAX_POINTS):
        if i < len(labeler.points):
            row.extend([labeler.points[i][0], labeler.points[i][1]])
        else:
            row.extend([None, None])
    return row


def main():
    image_paths = get_sorted_image_paths(FRAME_FOLDER)
    if not image_paths:
        print(f"No PNG images found in {FRAME_FOLDER.resolve()}")
        return

    labeler = Labeler()
    WINDOW_NAME = "Label Frame"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    data_rows = []

    for img_path in image_paths:
        frame_name = img_path.name
        image = cv2.imread(str(img_path))
        label_data = label_frame(image, frame_name, labeler, window_name=WINDOW_NAME)

        if label_data == "quit":
            print("User exited.")
            break
        elif label_data == "closed":
            print("Window was closed — recreating...")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            continue

        if label_data:
            data_rows.append(label_data)
            print(f"Labeled {frame_name}")
        else:
            print(f"Skipped {frame_name}")

    cv2.destroyAllWindows()

    # Build DLC-style multi-index header
    columns = [["frame"], [SCORER_NAME]]
    for part in BODY_PARTS:
        columns[0].extend([part, part])
        columns[1].extend(["x", "y"])

    header = pd.MultiIndex.from_arrays(columns)
    df = pd.DataFrame(data_rows, columns=header)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved labeled data to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
