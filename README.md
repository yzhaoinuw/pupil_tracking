![pupil_diameter_analysis_result_demo](https://github.com/user-attachments/assets/cca0794f-07d9-4ed1-a25e-310dd553e98d)

<p align="center"><em>Left: segmented pupil — Right: evolving pupil diameter plot</em></p>

# Pupil Analysis Pipeline

This script runs a full pipeline for **mouse pupil segmentation and size estimation** using a trained UNet model.  
You can start directly from a video file or from an existing folder of extracted frames. To obtain expected results, the video or images provided should have at least the majority of the eye contained in the 148 x 148 pixel - area in the center of the frames.  This is crucial to getting good results as the model was trained on 148 x 148 centered cropped images.  

---

## 📦 Installation
It is recommended that you first create a dedicated virtual environment, for example, with [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). Then in the virtual environment, navigate to a desired working directory and follow the steps below. 
### 1. Clone the repository
```bash
git clone https://github.com/yzhaoinuw/pupil_tracking.git
cd pupil_tracking
```
### 2. Install dependencies
```bash
pip install -e .
```
### 3. Download model checkpoint
Download `checkpoints/` from [Onedrive](https://uofr-my.sharepoint.com/:f:/g/personal/yzhao38_ur_rochester_edu/EkV9WCc9VQ1Mk1HUcfhZPhgBynbBr5YMczw2Stcv9aRFGQ?e=bndEXX) (contact Yue if you don't have access), and then place it in `pupil_tracking/`.


## 🏃 Basic Usage
After installation, you can run pupil analysis on a video like so
```bash
run-pupil-analysis --video_path /path/to/movie.avi
```

This will:

1. Extract evenly spaced frames from the video into a folder like `movie_frames/`
2. Run pupil segmentation and diameter estimation on those frames
3. Save the results (CSV + plot) into `movie_frames_result/`

---

## ⚙️ Key Arguments

| Argument            | Description                                                                                                       |
|---------------------|-------------------------------------------------------------------------------------------------------------------|
| `--video_path`      | Path to the input video file. If provided, frames will automatically be extracted before analysis.                |
| `--out_dir`         | Optional. Directory to save extracted frames. If not given, defaults to `<video_stem>_frames/` next to the video. |
| `--image_dir`       | Optional alternative to `--video_path`. Use this if you already have extracted PNG frames.                        |
| `--result_dir`      | Optional. Directory to save the CSV and plot outputs. If not given, defaults to `<image_dir>_result/`.            |
| `--output_mask_dir` | Optional. If provided, saves overlay images showing the predicted pupil mask blended onto the original frames.    |
| `--extraction_fps`  | Optional. Specifies the number of frames per second at which to extract the frames from the video (default: 5). If `--max_frames` is provided, and if the number of frames to be extracted at `--extraction_fps` would exceed `--max_frames`, then the actual `--extraction_fps` will be automatically reduced so that `--max_frames` number of frames will be extracted. |
| `--max_frames`      | Optional. Limits the maximum number of frames to extract from a video (default: 10,000). Useful for long recordings.        |
| `--pred_thresh`     | Optional. Ranging from 0 to 1, it specifies the confidence threshold for calssifying a pixel belonging to a pupil. For example, a value of 0.7 means that a pixel will be classified as a pupil pixel on if the model has a confidence that exceeds 0.7 that this pixel is a pupil pixel. Increase it to if the resulting segmentation overpredicts a pupil; reduce it if the resulting segmentation only finds part of a pupil. |  

---

## 💡 Examples

**From a video (auto frame extraction):**
```bash
run-pupil-analysis --video_path data/mouse1.avi
```

**From an existing folder of frames:**
```bash
run-pupil-analysis --image_dir data/mouse1_frames
```

**With custom output locations and segmentation masks:**
```bash
run-pupil-analysis \
  --video_path data/mouse1.avi \
  --out_dir data/frames_mouse1 \
  --result_dir data/results_mouse1 \
  --output_mask_dir data/masks_mouse1
```

---

## 📦 Output Files

After running, you’ll typically find:

| File | Description |
|------|--------------|
| `*_estimated_pupil_diameter.csv` | A table of estimated pupil diameters for each frame (in pixels). |
| `*_estimated_pupil_diameter.png` | A line plot showing pupil diameter over time (frame index on x-axis). |
| *(optional)* Mask images in `output_mask_dir` | PNGs with the pupil mask (in red) blended onto the original grayscale image. |

---

## 🧩 Typical Folder Structure

```
movie.avi
movie_frames/
    movie_00000.png
    movie_00001.png
    ...
movie_frames_result/
    movie_estimated_pupil_diameter.csv
    movie_estimated_pupil_diameter.png
```

---

## Developer Notes

### Model Training

#### Making Training Data
Create two folders in *pupil_tracking/*, *images_train/* and *masks_train/* if you haven't. Place your training images in *images_train/*. Once you have done this once, you can just add new training images to *images_train/*.
1. In Terminal/Anaconda Powershell Prompt, activate environment pupil_tracking, then run `labelme.exe`
to open the labelme interface to label images.
2. After you are done, **labelme** should have saved your labels as json files in *images_train/* along with your training images. Now run `python .\labelme_json2png.py`, which will create the masks (png files) and move them to *masks_train/*.
3. To create the validation set, create *images_validation/* and *masks_validation/* and then follow the same steps above, but remember to change **image_dir** and **mask_dir** in **labelme_json2png.py** accordingly.
4. To start training the model, run `python run_train.py`. You can modify the hyperparameters in **run_train.py** as needed.