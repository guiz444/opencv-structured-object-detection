# Vision-based Structured Object Detection and Pose Estimation

An end-to-end traditional computer vision system built with OpenCV, covering interactive color calibration, robust structured object detection, temporal stability filtering, and real-time 3D pose estimation using PnP.

---

## 🔍 Project Overview

This project implements a **complete vision pipeline** for detecting structured objects under constrained color conditions and estimating their 3D pose from monocular video.
The system is designed for **engineering-oriented vision tasks** such as robotics perception, industrial inspection, and embedded vision prototyping.

The pipeline avoids deep learning and instead focuses on **classical computer vision methods**, emphasizing interpretability, controllability, and real-time performance.

---

## 🧩 Pipeline Architecture

```
Input Video / Camera Stream
        ↓
HSV Color Calibration (Interactive)
        ↓
Color-based Segmentation (ROI Extraction)
        ↓
Geometric Shape Detection (Contours & Polygons)
        ↓
Temporal Filtering (Jump Suppression & Area Consistency)
        ↓
Corner Ordering & Keypoint Extraction
        ↓
PnP-based 3D Pose Estimation
        ↓
Real-time Visualization (Bounding Boxes & Axes)
```

---

## 📁 Repository Structure

```

```

---

## ⚙️ Key Features

### 1️⃣ Interactive HSV Calibration

* Real-time HSV threshold adjustment using trackbars
* Enables rapid adaptation to different lighting conditions
* Eliminates hard-coded color parameters

### 2️⃣ Robust Color-based ROI Extraction

* HSV segmentation + morphological operations
* Largest connected component selection for stable ROI
* Noise suppression through area filtering

### 3️⃣ Structured Shape Detection

* Contour detection with polygon approximation
* Aspect ratio and vertex-count constraints
* Distinguishes **squares vs arrow-like markers**

### 4️⃣ Temporal Stability Filtering

* Frame-to-frame centroid distance filtering
* Area consistency checks to suppress false positives
* Reduces jitter and sudden detection jumps

### 5️⃣ 3D Pose Estimation (PnP)

* Monocular pose estimation via `cv2.solvePnP`
* Predefined object coordinate system
* Real-time visualization with camera axes overlay

---

## 🧠 Techniques Used

* OpenCV (Python)
* HSV color space segmentation
* Morphological image processing
* Contour analysis & polygon approximation
* Temporal filtering (motion consistency)
* Perspective-n-Point (PnP) pose estimation

---

## 🚀 Applications

* Robotics perception and manipulation
* Industrial object localization
* Vision-based alignment and calibration
* Embedded / edge vision systems
* Computer vision teaching demonstrations

---

## 📸 Demo Outputs

* Real-time bounding box visualization
* Ordered corner keypoints
* 3D coordinate axes rendered on detected objects
* Video export of pose estimation results

---

## 📦 Environment

* Python ≥ 3.8
* OpenCV ≥ 4.5
* NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## 📝 Notes

* Camera intrinsic parameters are configurable
* The system is modular and can be extended to other colors or shapes
* Designed for real-time performance on standard CPUs

---

## 📄 License

This project is intended for academic, educational, and research use.
