# Smart Palletizer Challenge – Solution Summary

This document summarizes my complete solution to the NEURA Robotics Smart Palletizer challenge.
---
* Whole implementation was done within a devcontainer setup. By building the docker image you should have the same environment as I do. Also the docker image is generated via github action and can be pulled directly

## 1. 2D Box Detection – Synthetic Dataset + YOLO

### Synthetic Dataset Preparation

Due to limited real annotated data, I created a synthetic dataset using real box crops and background images to simulate various scenarios.

#### Steps:
1. **Crop Boxes**: Extracted box crops from real RGB images in `data/small_box` and `data/medium_box`.
2. **Collect Backgrounds**: Used various backgrounds from the `data/backgrounds/` folder.
3. **Overlay Crops Randomly**: Pasted cropped boxes onto backgrounds at random positions, scales, and sometimes with small rotations.
4. **Save in YOLO Format**: Saved the images and label files compatible with YOLO format for training.

This process is detailed in the notebook:
`notebooks/A1_prepare_training_yolo.ipynb`

#### Example – Synthetic Training Sample

![Synthetic Training Data](docs/imgs/synthetic_training_data.jpg)

This method allowed me to create hundreds of varied training examples that improved generalization for YOLO.

---

### YOLO Model Training
- Trained a YOLOv8 model using the synthetic dataset.
- Converted annotations to YOLO format (bounding boxes normalized between 0–1).
- Trained in Google Colab for faster iteration.

Training process notebook: `notebooks/A1_Train_YOLO_Model.ipynb`

### Inference with Masked YOLO
- The `MaskedYOLODetector` class masks out irrelevant regions using depth filtering before running YOLO.
- Helps reduce false positives.

Detection demo: `notebooks/A1_run_yolo_detector.ipynb`

![YOLO Detection Output](docs/imgs/yolo_detection.png)

---

## 2. Planar Patch Detection (3D)

- Used Open3D to convert depth images into point clouds.
- Applied plane segmentation with RANSAC and filtered patches based on area and orientation.

Demo notebook: `notebooks/A2_run_patch_detector.ipynb`

Examples:
![Planar Patch Detection 1](docs/imgs/planar1.png)
![Planar Patch Detection 2](docs/imgs/planar2.png)

---

## 3. Point Cloud Cleaning

- Downsampled with voxel grid
- Removed outliers with statistical methods
- Fitted planes using RANSAC and filtered noise

Notebook: `notebooks/A3_clean_point_clouds.ipynb`


---

## 4. Box Pose Estimation (6D)

Estimated 6D poses from top planar surfaces.
- Used known dimensions and orientation of planar patches
- Identified if a patch is a top, front, or side face
- Computed center and rotation matrix
- Transformed from camera to world frame

Notebook: `notebooks/A4_estimate_box_pose.ipynb`

![Pose Estimation](docs/imgs/rotation_estimation.png)

---

## Pipeline Overview

Modular architecture:
- `MaskedYOLODetector`: 2D detection with mask filtering
- `PlanarPatchDetector`: 3D planar patch extraction
- `BoxPoseEstimator`: computes 6D poses

Pipeline diagram:
![Pipeline](docs/imgs/pipeline.png)

---

## ROS Integration

ROS nodes prepared for real-time pipeline execution:

```bash
# Terminal 1: ROS bag playback
rosbag play <path_to_rosbag> --loop --clock

# Terminal 2: Box detection
roslaunch smart_palletizer box_detector.launch

# Terminal 3: Planar patch node
roslaunch smart_palletizer planar_patch_detector.launch

# Terminal 4: Visualize
rqt_image_view
```

---

## Documentation

All code is documented and compiled with Sphinx.

To regenerate HTML docs:
```bash
cd docs
make html SPHINXBUILD="python3 /opt/conda/envs/rosenv/bin/sphinx-build"
```

Output available in `docs/build/html/index.html`

## Code Overview:
**notebooks/:** contain Jupyter notebooks for each step of the pipeline

**lib/:** contains the main classes and functioons used both by the notebooks and the ROS nodes

**src/scripts/:** contains the ROS nodes

**data/backgrounds/:** contains the background images used to create the synthetic dataset

**data/model:** contains the trained YOLOv8 model

**data/dataset:** contains the synthetic dataset. (Only first 20 images are saved to make this repo small. For training during synthertic data generation around 1000 dataset will
be generated
