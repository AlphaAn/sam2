# SAM 2 Toolkit: Detailed Reproduction Guide

**Author:** Thai Khac Duc An

**Project:** Segment Anything Model 2 (SAM 2) Integration & Testing pipeline

This folder contains a suite of Python scripts designed to test, analyze, and utilize Meta's Segment Anything Model 2 (SAM 2) for zero-shot segmentation and tracking. 

Below are detailed instructions on how to run each tool, alongside an explanation of how they work and what outputs they generate.

---

## 1. Environment Setup
Before running any scripts, you must activate the lab's dedicated Python environment that contains PyTorch and the SAM 2 dependencies. Open your terminal/command prompt, navigate to this folder, and run:
```bash
# Activate the environment
conda activate sam2
cd sam2/notebooks
```
*(Note: All scripts will automatically detect your hardware and utilize CUDA + bfloat16 precision if a compatible GPU is available, or fallback to CPU if not).*

---

## 2. Tool Descriptions & Execution Commands

### A. System Profiling & Data Validation (`task1.py`)
**Description:** Before running massive datasets through SAM 2, you need to know the absolute limits of the lab's hardware. This script serves two purposes: First, it scans the `test/input/` directory to verify that SAM 2 can successfully ingest your specific image formats (e.g., `.tiff`, `.jpg`) and video formats. Second, it runs a dynamic Out-Of-Memory (OOM) stress test. It feeds progressively larger, dummy high-resolution images into the model until the GPU crashes, catching the error and printing the exact maximum spatial resolution the current machine can handle safely.
* **Command:**
  ```bash
  python task1.py
  ```
* **Output:** Terminal logs detailing compatible files and the maximum safe resolution boundary for the current GPU.

### B. Internal Encoder Feature Analysis (`task2.py`)
**Description:** This tool acts as an "x-ray" for the AI. Instead of generating final segmentation masks, it intercepts the model's internal forward pass. It extracts the raw, high-dimensional feature maps from SAM 2's Hiera encoder and collapses them into 2D heatmaps. This is highly useful for researchers who want to see exactly how the model is spatializing and "understanding" an image conceptually before it makes decisions.
* **Command:**
  ```bash
  python task2.py test/input/sample.jpg
  ```
* **Output:** A Matplotlib window displaying the original image alongside the multi-scale, color-coded activation heatmaps.

### C. Interactive Single-Image Segmentation (`task3.py`)
**Description:** This script provides a user-friendly Graphical User Interface (GUI) for manual segmentation. It leverages SAM 2's `multimask_output` capability. Because a single point click can be ambiguous (e.g., clicking a tire—do you mean the tire, the whole car, or the car and the road?), the model generates three potential masks and ranks them by its own internal confidence score (Predicted IoU).
* **Command:**
  ```bash
  python task3.py test/input/sample.jpg
  ```
* **Controls:**
  * `Left-Click`: Add a positive (Green) point to include a feature.
  * `Right-Click`: Add a negative (Red) point to exclude background.
  * `ENTER` or `q`: Submit the prompts to the model.
* **Output:** A Matplotlib window showing the original image with your click coordinates, alongside the Top 3 predicted masks overlaid in blue.

### D. Automatic Whole-Image Segmentation (`auto_segment_img.py`)
**Description:** This tool performs zero-shot object discovery. Instead of waiting for user clicks, it utilizes the `SAM2AutomaticMaskGenerator` to flood the entire image with a grid of point prompts. It resolves ambiguities, filters out duplicates, and detects every distinct object or feature it can find in the image.
* **Command:**
  ```bash
  python auto_segment_img.py test/input/sample.jpg
  ```
* **Output:** A Matplotlib window displaying the original image covered in distinct, randomly assigned color overlays for every single object the model discovered.

### E. Interactive Multi-Object Video Tracking (`interactive_segment_video.py`)
**Description:** This is an end-to-end temporal tracking pipeline. It extracts the first frame of a provided video (or folder of frames) and opens the interactive GUI. Users can define multiple different objects in that first frame. The script then feeds these spatial prompts into SAM 2's video predictor, leveraging its temporal memory to track those specific objects across all subsequent frames, even handling occlusions.
* **Command:**
  ```bash
  python interactive_segment_video.py test/input/sample_video.mp4
  ```
* **Controls:**
  * `Left/Right Click` to define Object 1.
  * Press `n` on your keyboard to lock in Object 1, and begin clicking to define Object 2 (repeat for as many objects as you want).
  * Press `ENTER` to begin tracking.
* **Output:** Automatically renders and saves an `[input_name]_output.mp4` file in the same directory, featuring color-coded mask overlays and ID tags for your tracked objects.

### F. Fully Automatic Video Tracking (`auto_segment_video.py`)
**Description:** The ultimate pipeline tool. This script merges the capabilities of the automatic image segmenter with the video temporal tracker. It automatically detects *every* distinct object in the first frame of your video, assigns them all unique IDs, and then tracks dozens of objects simultaneously through the video. Because tracking 50+ objects requires massive VRAM, this script utilizes dynamic CPU-offloading to prevent the GPU from crashing during rendering.
* **Command:**
  ```bash
  python auto_segment_video.py test/input/sample_video.mp4
  ```
* **Output:** Saves a fully annotated `[input_name]_output.mp4` video file, where every detected object in the scene is highlighted and tracked automatically.

---
**Need to save disk space?**

By default, the video scripts (`interactive_segment_video.py` and `auto_segment_video.py`) will save the final `.mp4` *and* a folder containing every individually tracked `.jpg` frame. If you are processing a massive video and only want the `.mp4`, append the `--no_frames` flag to your command:
```bash
python auto_segment_video.py test/input/sample_video.mp4 --no_frames
```

---

## 3. Past & Reference Results

If you want to see examples of what the final outputs (segmented images, heatmaps, or tracked `.mp4` videos) should look like without running the code yourself, you can find past runs and reference results stored in the following directories:
* **`test/` folder:** Contains the raw sample input files as well as some baseline test outputs.
* **`results/` folder:** Contains the finalized outputs from previous runs for quick reference.