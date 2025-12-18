# Vision-based Structured Object Detection and Pose Estimation

## 🇬🇧 English Version

An **engineering-oriented, end-to-end traditional computer vision pipeline** built with OpenCV, designed for **robust structured object detection and monocular 3D pose estimation** under constrained color conditions.

The system emphasizes **interpretability, controllability, and real-time performance**, making it suitable for robotics perception, industrial inspection, and embedded vision prototyping scenarios where deep learning approaches may be impractical or unnecessary.

---

## 🔍 Project Overview

This project implements a **complete classical vision processing pipeline** for detecting structured objects and estimating their 3D pose from monocular video streams.

Unlike deep-learning-based solutions, the pipeline relies entirely on **traditional computer vision techniques**, allowing fine-grained parameter control, predictable behavior, and efficient CPU-only deployment.

---

## 🧠 Design Philosophy

This project deliberately avoids deep learning and focuses on **classical computer vision methods** for the following reasons:

* Strong interpretability and algorithmic transparency
* Low computational overhead suitable for real-time CPU execution
* Ease of debugging and system-level tuning
* Applicability to structured environments with clear geometric and color priors

The entire system is designed as a **modular vision pipeline**, where each stage can be independently tested, tuned, or replaced.

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
1_hsv_calibration/
├── hsv_estimate.py
├── hsv_exact.py
├── split.py
├── assets/
└── README.md

2_structured_detection/
├── v1_horizon.py
├── v2_polygon.py
├── v3_spin.py
├── assets/
└── README.md

3_application_video/
├── video_detector.py
├── output_arrow_detect_v3.mp4
├── demo.gif
└── README.md

4_pose_estimation_pnp/
├── detect_squares_area_filter/          # Square-based pose estimation with area stability filtering
│   ├── imagepoints.py                   # Corner extraction & ordering
│   ├── pnp.py                           # Core solvePnP pipeline
│   ├── pnp_math.py                      # Pose math utilities & smoothing
│   └── pnp_Matplotlib3D.py              # 3D pose visualization
├── other_vision/
|    ├── detect_quad_from_hex_combo/       # Polygon-combination-based pose estimation (1 square + 3 hex)
|    │   ├── test.py
|    │   └── test0.py
|    └── gold_silver_detector.py           # Real-time ore detection (no solvePnP)
├──assets/stone.mp4
└──README.md


```

---

## ⚙️ Key Features

### 1️⃣ Interactive HSV Color Calibration

* Support multi-point mouse sampling to count color distribution and complete real-time HSV threshold adjustment
* Rapid adaptation to different lighting conditions
* Eliminates hard-coded color parameters

### 2️⃣ Robust Color-based ROI Extraction

* HSV segmentation combined with morphological operations
* Largest connected component selection for stable ROI extraction
* Noise suppression via area-based filtering

### 3️⃣ Structured Shape Detection

* Contour extraction with polygon approximation
* Geometric constraints based on vertex count and aspect ratio
* Differentiates structured targets such as squares and arrow-like markers

### 4️⃣ Temporal Stability Filtering

* Frame-to-frame centroid distance constraints to suppress sudden jumps
* Area consistency checks to filter transient false positives
* Significantly improves detection stability in continuous video streams

### 5️⃣ 3D Pose Estimation with Multi-Strategy PnP

* Monocular 6-DoF pose estimation using `cv2.solvePnP`
* Supports multiple geometric strategies:
  - Square-based minimum bounding rectangle detection
  - Polygon-combination-based pose inference (1 square + 3 hex)
* Automatic 2D–3D corner correspondence construction
* Temporal smoothing and jump suppression for stable pose output
* Real-time visualization with projected 3D coordinate axes

---

## 🔬 Pose Estimation Module Details (4_pose_estimation_pnp)

The pose estimation module is designed as an **independent, strategy-driven subsystem** within the overall vision pipeline.

Instead of relying on a single detection pattern, the module provides **multiple geometric pose inference strategies**, enabling robust deployment across different structured scenarios.

### Implemented Strategies

- **Square + Area Stability Filtering**
  - Uses minimum bounding rectangles
  - Rejects unstable detections via area similarity constraints
  - Suitable for planar fiducial-like targets

- **Polygon Combination Pose Inference**
  - Infers a valid quadrilateral from a combination of polygons (1 square + 3 hexagons)
  - Uses geometric scoring and reprojection validation
  - Designed for partially structured or composite markers

- **Real-time Detection without Pose Solving**
  - Fast color + shape-based detection
  - Designed for monitoring and classification tasks where pose is unnecessary

Each strategy can be executed independently and integrated into the full pipeline when required.
Key pose stability parameters (e.g. jump threshold, area consistency ratio, axis scale) are documented in the module-level README under `4_pose_estimation_pnp/`.

---

## 🧠 Techniques Used

* OpenCV (Python)
* HSV color space segmentation
* Morphological image processing
* Contour analysis & polygon approximation
* Temporal filtering and motion consistency constraints
* Perspective-n-Point (PnP) pose estimation

---

## 🚀 Applications

* Robotics perception and manipulation
* Industrial object localization and alignment
* Vision-based calibration systems
* Embedded and edge vision platforms
* Computer vision education and demonstrations

---

## 📸 Demo Outputs

* Real-time bounding box and contour visualization
* Ordered corner keypoints overlay
* 3D coordinate axes rendered on detected objects
* Video export of pose estimation results

---

## 📦 Environment

* Python ≥ 3.8
* OpenCV ≥ 4.5
* NumPy

```bash
pip install opencv-python numpy
```

---

## 📄 License

This project is intended for **academic, educational, and research use**.

---

# 🇨🇳 中文版本

## 项目简介

本项目是一个**面向工程应用的端到端传统计算机视觉系统**，基于 OpenCV 构建，用于在**颜色受限场景下实现结构化目标检测与单目三维位姿估计**。

系统强调 **可解释性、可控性与实时性能**，适用于机器人感知、工业检测以及嵌入式视觉原型开发等不依赖深度学习的应用场景。

---

## 🔍 项目概述

该项目实现了一套完整的**传统计算机视觉处理流水线**，能够从单目视频流中完成结构化目标的稳定检测与三维位姿估计。

与基于深度学习的方法不同，本系统完全基于**经典视觉算法**，具备参数可控、行为可预测、部署成本低等优势，适合在 CPU 平台上实时运行。

---

## 🧠 设计理念

本项目有意避免使用深度学习方法，而专注于传统计算机视觉技术，主要基于以下考虑：

* 算法行为清晰，可解释性强
* 计算开销低，适合实时与嵌入式场景
* 调试与工程调参成本低
* 适用于具有明确几何与颜色先验的结构化环境

整体系统采用**模块化视觉流水线设计**，各处理阶段相互解耦，便于独立调试、替换与扩展。

---

## 🧩 系统流水线结构

```
视频输入 / 摄像头
        ↓
HSV 颜色交互式标定
        ↓
基于颜色的区域分割（ROI）
        ↓
几何结构检测（轮廓 / 多边形）
        ↓
时序稳定性过滤（跳变抑制 + 面积一致性）
        ↓
角点排序与关键点提取
        ↓
基于 PnP 的三维位姿估计
        ↓
实时结果可视化（检测框 & 三维坐标轴）
```

---

## 📁 仓库结构说明

仓库按功能阶段划分，每个子模块对应视觉流水线中的一个关键步骤，并包含独立说明文档。
1_hsv_calibration/          # HSV 标定模块
├── hsv_estimate.py
├── hsv_exact.py
├── split.py
├── assets/
└── README.md

2_structured_detection/      # 结构化检测模块
├── v1_horizon.py
├── v2_polygon.py
├── v3_spin.py
├── assets/
└── README.md

3_application_video/         # 视频应用模块
├── video_detector.py
├── output_arrow_detect_v3.mp4
├── demo.gif
└── README.md

4_pose_estimation_pnp/       # PnP 位姿估计模块
├── detect_squares_area_filter/          # 基于面积稳定性过滤的方形位姿估计
│   ├── imagepoints.py                   # 角点提取与排序
│   ├── pnp.py                           # 核心 solvePnP 流程
│   ├── pnp_math.py                      # 位姿数学工具与平滑处理
│   └── pnp_Matplotlib3D.py              # 三维位姿可视化
├── other_vision/
|    ├── detect_quad_from_hex_combo/      # 基于多边形组合的位姿推理（1方 + 3六边形）
|    |   ├── test.py
|    |   └── test0.py
|    └── gold_silver_detector.py          # 实时矿石检测（不含 solvePnP）
├──assets/stone.mp4
└──README.md
---

## ⚙️ 核心功能

### 1️⃣ HSV 颜色交互式标定

* 支持鼠标多点采样统计颜色分布，完成实时 HSV 阈值调整
* 快速适配不同光照条件
* 避免硬编码颜色参数

### 2️⃣ 稳健的颜色区域提取

* HSV 分割结合形态学操作
* 通过连通域分析提取稳定 ROI
* 基于面积阈值的噪声抑制

### 3️⃣ 结构化目标几何检测

* 基于轮廓检测与多边形近似
* 利用顶点数、长宽比等几何约束
* 区分方形、箭头等结构化标记

### 4️⃣ 时序稳定性过滤

* 基于质心帧间距离的跳变抑制
* 面积一致性约束过滤瞬时误检
* 显著提升视频流检测稳定性

### 5️⃣ 基于多策略 PnP 的三维位姿估计
* 使用 cv2.solvePnP 实现单目 6-DoF 位姿求解
* 支持多种几何策略：
* 基于方形的最小外接矩形检测
* 基于多边形组合（1方 + 3六边形）的位姿推理
* 自动构建 2D–3D 角点对应关系
* 时序平滑处理与跳变抑制，确保位姿输出稳定
* 实时投影三维坐标轴进行可视化

---

## 🔬 位姿估计模块细节 (4_pose_estimation_pnp)

位姿估计模块被设计为视觉流水线中一个**独立、由策略驱动的子系统**。

该模块不依赖单一检测模式，而是提供**多种几何位姿推理策略**，确保在不同结构化场景下的稳健部署。

### 已实现的策略

* **方形 + 面积稳定性过滤**
* 使用最小外接矩形
* 通过面积相似度约束剔除不稳定检测
* 适用于平面类基准目标（Fiducial targets）


* **多边形组合位姿推理**
* 从多边形组合（1个方形 + 3个六边形）中推导有效四边形
* 使用几何评分与重投影验证
* 专为局部结构化或复合标记设计


* **不含位姿解算的实时检测**
* 极速颜色 + 形状检测
* 专为无需位姿信息的监控和分类任务设计



每种策略均可独立执行，并在需要时集成到完整流水线中。关键的位姿稳定性参数（如跳变阈值、面积一致性比例、坐标轴缩放）记录在 `4_pose_estimation_pnp/` 的模块级 README 中。

---

## 🧠 使用技术

* OpenCV（Python）
* HSV 颜色空间分割
* 形态学图像处理
* 轮廓分析与多边形拟合
* 时序一致性过滤算法
* PnP 三维位姿估计

---

## 🚀 应用场景

* 机器人环境感知与抓取定位
* 工业目标定位与对准
* 视觉标定与定位系统
* 嵌入式 / 边缘视觉平台
* 计算机视觉教学与演示

---

## 📦 运行环境

* Python ≥ 3.8
* OpenCV ≥ 4.5
* NumPy

---

## 📄 许可说明

本项目面向**学术研究、教学与科研用途**。

