
# Structured Color Target Detection Module (Single-Frame)

*Geometry-Constrained Structured Color Target Detection*

---

## Module Overview

This module focuses on **color segmentation + contour analysis + geometric bounding**, without handling video streams, interactive control, or system scheduling.
It serves as a bridge between low-level color extraction and higher-level temporal filtering or pose estimation.

Although the current examples use **blue arrow light strips** as reference targets, the detection logic itself is **independent of target semantics**.
By adjusting HSV parameters (from the HSV calibration module) and geometric rules, this module can be generalized to **other color-consistent, structurally stable targets**.

The design emphasizes **interpretability, controllability, and geometric stability**, relying on classical computer vision methods suitable for real-time and embedded scenarios.

---

## Design Objectives

* Achieve stable detection under color constraints
* Apply explicit geometric rules instead of black-box models
* Produce deterministic, reproducible results
* Integrate seamlessly with downstream video processing or pose estimation modules

---

## Detection Workflow

```
HSV Color Segmentation
        ↓
Morphological Processing
        ↓
Contour Extraction
        ↓
Polygon Approximation
        ↓
Geometric Filtering
        ↓
Single-Frame Candidate Output
```

---

## Detection Strategies Overview

Three contour bounding strategies are implemented for comparing detection accuracy, stability, and applicability:

| Version | Method                                         | Features                                 | Suitable Scenarios                                   |
| ------- | ---------------------------------------------- | ---------------------------------------- | ---------------------------------------------------- |
| v1      | `boundingRect` (horizontal rectangle)          | Fast and simple                          | Nearly horizontal targets, real-time priority        |
| v2      | `approxPolyDP` + `boundingRect`                | Smoother contours, reduces over-bounding | Slightly irregular contours                          |
| v3      | `minAreaRect` (minimum area rotated rectangle) | Fits any angle, most compact bounding    | Targets with significant tilt or complex orientation |

---

## Example Input Image

The following image shows the **same input** for comparing different bounding strategies:

![Input Image](assets/img2.png)

---

## Detection Results Comparison

### v1 — Horizontal Rectangle (Bounding Rect)

Uses raw contours to generate horizontal bounding rectangles. Computationally cheap but may not tightly fit tilted targets.

![v1 Detection Result](assets/arrow_detect_v1_20251012_145828.png)

---

### v2 — Polygon Approximation + Horizontal Rectangle

First applies `approxPolyDP` to simplify contours, then generates bounding rectangles.
This reduces the influence of edge noise on bounding results.

```python
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)
```

![v2 Optimized Result](assets/arrow_poly_v2_20251012_194948.png)

Adjusting the approximation accuracy produces smoother contours.

---

### v3 — Minimum Area Rotated Rectangle (Min Area Rect)

Generates a minimum-area rotated rectangle, accurately fitting arrow strips at any angle.
This is the **highest-precision and most versatile** approach in the module.

![v3 Detection Result](assets/arrow_detect_v3_20251012_145841.png)

---

## Single-Frame Detection Processing Steps

1. **Read image and convert color space**

```python
img = cv2.imread(img_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

2. **Segment target color regions**

```python
lower_blue = np.array([90, 30, 180], dtype=np.uint8)
upper_blue = np.array([140, 255, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
```

3. **Morphological closing to reduce noise**

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
```

4. **Contour detection**

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

5. **Target filtering and bounding**

* Filter out small noise contours
* Generate bounding boxes according to the strategy version
* Draw and save results

---

## Core Code Examples

### v1 — Raw Contour + Horizontal Rectangle

```python
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
```

### v2 — Polygon Approximation + Horizontal Rectangle

```python
epsilon = 0.03 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
x, y, w, h = cv2.boundingRect(approx)
cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
```

### v3 — Minimum Area Rotated Rectangle

```python
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect).astype(int)
cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
```

---

## Role in the Overall Pipeline

```
HSV Calibration Module
        ↓
Structured Color Target Detection  ← (This Module)
        ↓
Temporal Filtering / Tracking
        ↓
PnP Pose Estimation
```

This module works purely on single-frame information and can be used both for images and seamlessly integrated into video pipelines.

---

## Typical Applications

* Structured color marker detection
* Industrial vision indicator localization
* Front-end processing for pose estimation systems
* Teaching examples for geometric computer vision

---

## Generalization Notes

* Target color is configured via upstream HSV calibration module
* Geometric rules can be flexibly adjusted for different targets
* Core logic is independent of arrow or any specific semantic

---

## Module Usage Example

```python
from v3_spin import detect_blue_light_arrow

mask, result_img, results = detect_blue_light_arrow("img2.png")
```

**Output explanation:**

* `mask`: binary mask of target color regions
* `result_img`: annotated detection image
* `results`: geometric information of detected targets

---

## Design & Optimization Highlights

* **Color robustness**: HSV thresholds can be tuned for different lighting conditions
* **Noise suppression**: area filtering + morphological operations reduce false positives
* **Geometric adaptability**: minimum-area rotated rectangles handle multi-directional targets
* **Modular decoupling**: interface design allows easy integration into video or system-level applications

---

## Why Polygon Approximation (`approxPolyDP`) Is Used

Although `findContours` extracts full boundary points, raw contours often contain many small points, which are unsuitable for geometric analysis and stable bounding.

`approxPolyDP` geometrically simplifies the contour and provides:

* Edge jitter removal for more stable contours
* Reduced computational complexity for downstream geometric processing
* Structured contours for tighter bounding box generation

In structured color target detection, polygon approximation is **not only for visual smoothing but also a fundamental step for structured target analysis**.

---


# 结构化颜色目标检测模块（单帧）

*基于几何约束的结构化颜色目标检测*

---

## 模块概览

该模块聚焦于**颜色分割 + 轮廓分析 + 几何框选**，不涉及视频读取、交互控制或系统调度。
它用于连接低层颜色提取与上层的时序滤波或位姿估计。

虽然当前示例使用 **蓝色箭头灯条** 作为参考目标，但检测逻辑本身 **与目标语义无关**。
通过调整 HSV 参数（由 HSV 标定模块提供）和几何规则，该模块可泛化应用于 **其他颜色一致、结构稳定的目标**。

模块设计强调 **可解释性、可控性与几何稳定性**，采用传统计算机视觉方法，适用于实时与嵌入式场景。

---

## 设计目标

* 在颜色约束条件下实现稳定检测
* 使用显式几何规则，而非黑盒模型
* 行为确定、结果可复现
* 可与下游视频处理或位姿估计模块无缝衔接

---

## 检测流程

```
HSV 颜色分割
        ↓
形态学处理
        ↓
轮廓提取
        ↓
多边形近似
        ↓
几何约束筛选
        ↓
单帧候选目标输出
```

---

## 检测策略概览

模块内部实现三种轮廓框选策略，用于对比检测精度、稳定性与适用场景：

| 版本 | 方法                              | 特点            | 适用场景           |
| -- | ------------------------------- | ------------- | -------------- |
| v1 | `boundingRect` 水平矩形             | 速度快、实现简单      | 目标近似水平，对实时性要求高 |
| v2 | `approxPolyDP` + `boundingRect` | 轮廓更平滑，减少过度框选  | 轮廓略不规则的目标      |
| v3 | `minAreaRect` 最小旋转矩形            | 可匹配任意角度，框选最紧凑 | 目标存在明显倾斜的复杂场景  |

---

## 示例输入图像

以下示例展示了**同一输入图像**下不同框选策略的效果对比：

![原图示例](assets/img2.png)

---

## 不同策略的检测结果对比

### v1 — 水平矩形框（Bounding Rect）

使用原始轮廓生成水平矩形框，计算开销小，但对倾斜目标包围不够紧密。

![v1 检测结果](assets/arrow_detect_v1_20251012_145828.png)

---

### v2 — 多边形近似 + 水平矩形框

先使用 `approxPolyDP` 对轮廓进行几何简化，再生成矩形框。
这减少了边缘噪声对框选结果的影响。

```python
epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 2)
```

![v2 优化后结果](assets/arrow_poly_v2_20251012_194948.png)

调节近似精度可获得更平滑轮廓。

---

### v3 — 最小旋转矩形（Min Area Rect）

生成最小旋转矩形，可准确拟合任意角度的箭头灯条。
这是模块中**精度最高、适用性最强**的策略。

![v3 检测结果](assets/arrow_detect_v3_20251012_145841.png)

---

## 单帧检测处理步骤

1. **读取图像并转换颜色空间**

```python
img = cv2.imread(img_path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

2. **目标颜色区域分割**

```python
lower_blue = np.array([90, 30, 180], dtype=np.uint8)
upper_blue = np.array([140, 255, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_blue, upper_blue)
```

3. **形态学闭操作去噪**

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
```

4. **轮廓检测**

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

5. **目标筛选与框选**

* 过滤面积过小的噪声轮廓
* 根据策略版本生成对应包围框
* 绘制并保存结果

---

## 核心代码示例

### v1 — 原始轮廓 + 水平矩形

```python
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
```

### v2 — 多边形近似 + 水平矩形

```python
epsilon = 0.03 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
x, y, w, h = cv2.boundingRect(approx)
cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
```

### v3 — 最小旋转矩形

```python
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect).astype(int)
cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)
```

---

## 在整体系统中的作用

```
HSV 标定模块
        ↓
结构化颜色目标检测  ←（本模块）
        ↓
时序滤波 / 跟踪
        ↓
PnP 位姿估计
```

本模块仅依赖单帧信息，可用于图像处理，也可无缝集成到视频管线。

---

## 典型应用场景

* 结构化颜色标记检测
* 工业视觉指示元件定位
* 位姿估计系统前端处理
* 几何视觉方法教学示例

---

## 泛化说明

* 目标颜色由上游 HSV 标定模块配置
* 几何规则可灵活调整以适应不同目标
* 核心逻辑不依赖箭头或特定语义

---

## 模块调用示例

```python
from v3_spin import detect_blue_light_arrow

mask, result_img, results = detect_blue_light_arrow("img2.png")
```

**输出说明：**

* `mask`：目标颜色二值掩膜
* `result_img`：标注检测结果的图像
* `results`：检测到的目标几何信息

---

## 设计与优化要点

* **颜色鲁棒性**：HSV 阈值可根据光照条件调节
* **噪声抑制**：面积过滤 + 形态学操作降低误检
* **几何适配性**：最小旋转矩形可稳定适配多方向目标
* **模块解耦**：接口设计便于集成到视频或系统级应用

---

## 附：为何使用多边形近似（`approxPolyDP`）

虽然 `findContours` 能提取完整边界点，但原始轮廓通常包含许多细碎点，不利于几何分析和稳定框选。

`approxPolyDP` 对轮廓进行几何简化，可：

* 消除边缘抖动，提高轮廓稳定性
* 降低计算复杂度，便于后续几何处理
* 提供结构化轮廓，实现更紧凑的包围框

在结构化颜色目标检测中，多边形近似不仅用于视觉平滑，更是**结构化目标分析的基础步骤**。

---

