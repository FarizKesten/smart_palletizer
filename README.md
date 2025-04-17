# Smart Palletizer - Solution Summary

## 1. 2D Box Detection

### Approach
- Used synthetic training data created from real box templates.
- Generated diverse backgrounds and inserted box crops at various positions.
- Trained a YOLOv8 model on this synthetic dataset.

### Components
- `notebooks/A1_prepare_training_yolo.ipynb`: Synthetic data generation.
- `notebooks/A1_Train_YOLO_Model.ipynb`: YOLO model training.
- `notebooks/A1_run_yolo_detector.ipynb`: Detection visualization.

### Result
- Accurate 2D detection of small and medium boxes using color images and YOLOv8.
- Used depth thresholding to mask irrelevant regions.

---

## 2. Planar Patch Detection (3D)

### Approach
- Used Open3D to convert depth images into point clouds.
- Applied normal filtering and RANSAC to isolate flat surfaces.
- Grouped planar patches per detected box.

### Components
- `notebooks/A2_run_patch_detector.ipynb`: End-to-end planar patch detection.
- Detected patches filtered by size, orientation, and alignment.

### Result
- Robust detection of top box surfaces in cluttered 3D environments.

---

## 3. Point Cloud Post-Processing

### Approach
- Voxel downsampling to reduce noise.
- Statistical outlier removal.
- Plane segmentation to filter valid surfaces.

### Components
- `notebooks/A3_clean_point_clouds.ipynb`

### Result
- Clean and compact point clouds, preserving box dimensions.

---

## 4. Box Pose Estimation (6D)

### Approach
- Used only top planar patch for stability.
- Estimated rotation matrix based on known face orientation.
- Computed world coordinates using extrinsics.

### Components
- `notebooks/A4_estimate_box_pose.ipynb`
- `BoxPoseEstimator` class implementation.

### Result
- Accurate 6D pose estimation in the world frame.

---

## Pipeline Components

- `MaskedYOLODetector`: YOLO inference and mask filtering.
- `PlanarPatchDetector`: Point cloud segmentation and patch detection.
- `BoxPoseEstimator`: Pose estimation from patches and box class.

### Pipeline Diagram
![Pipeline](docs/imgs/pipeline.png)

---

## ROS Integration

### Commands
```bash
rosbag play <path_to_rosbag> --loop --clock
roslaunch smart_palletizer box_detector.launch
roslaunch smart_palletizer planar_patch_detector.launch
rqt_image_view
```

---

## Documentation

- Python docstrings used throughout the implementation.
- Sphinx documentation generated under `docs/`.
- Viewable locally with `make html`.

---

## Visual Examples

### YOLO Detection:
![YOLO Detection Output](docs/imgs/yolo_detection.png)

### Planar Patches:
![Planar Patch Detection 1](docs/imgs/planar1.png)
![Planar Patch Detection 2](docs/imgs/planar2.png)

### Pose Estimation:
![Pose Estimation](docs/imgs/rotation_estimation.png)
