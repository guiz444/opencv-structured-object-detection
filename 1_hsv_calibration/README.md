
# HSV Calibration Tools

*Interactive HSV Threshold Selection for Vision Pipelines*

## Overview

This module provides **interactive HSV calibration utilities** for computer vision systems.
It is designed to **determine reliable color thresholds** before applying downstream detection, geometry filtering, or pose estimation.

The module is intentionally lightweight and decoupled, focusing purely on **parameter selection and validation**.

---

## Key Components

### 1. Exact HSV Picker

* Multi-point mouse sampling on target pixels
* Computes **tight HSV bounds** using min / max statistics
* No tolerance expansion applied

**Typical usage:**

* Controlled illumination
* High color consistency
* Scenarios requiring strict false-positive suppression

---

### 2. Estimating HSV Picker (Tolerance Mode)

* Multi-point sampling with **adaptive tolerance expansion**
* Default tolerance:

  * Hue ±10
  * Saturation ±40
  * Value ±40
* Real-time mask overlay for visual feedback

**Typical usage:**

* Illumination variation
* Thin or reflective targets (e.g. light bars)
* Practical deployment environments

---

### 3. Mask Application Example

Demonstrates how calibrated HSV ranges can be directly applied to:

* Binary mask generation
* Color-based region extraction
* Result persistence for debugging and evaluation

---

## Installation

```bash
pip install opencv-python numpy
```

---

## Usage

### Exact Mode

```python
from 1_hsv_calibration.hsv_exact import hsv_mask_exact_tool
lower_hsv, upper_hsv = hsv_mask_exact_tool("img1.png")
```

### Estimating Mode

```python
from 1_hsv_calibration.hsv_estimate import hsv_mask_estimating_tool
lower_hsv, upper_hsv = hsv_mask_estimating_tool("img1.png")
```

---

## Design Comparison

| Mode       | Tolerance | Strength   | Recommended Scenario  |
| ---------- | --------- | ---------- | --------------------- |
| Exact      | None      | Precision  | Controlled lighting   |
| Estimating | Adaptive  | Robustness | Real-world conditions |

---

## Returned Values

* `lower_hsv`, `upper_hsv`: HSV boundaries `[H, S, V]`
* Returns `(None, None)` if no pixels are selected

---

## Role in the Pipeline

```
HSV Calibration
      ↓
Color Segmentation
      ↓
Geometric Filtering
      ↓
Pose Estimation / Tracking
```

---

---

# HSV 标定工具

*用于视觉系统的交互式颜色阈值选取模块*

## 模块说明

本模块提供一组 **交互式 HSV 颜色阈值标定工具**，用于视觉系统中颜色分割前的参数确定。

该模块仅关注 **HSV 参数选取与验证**，与具体目标检测或位姿估计逻辑解耦，便于在不同实验和任务中复用。

---

## 功能组成

### 1. 精确 HSV 选取（Exact Mode）

* 通过鼠标多点点击采样目标颜色
* 使用最小 / 最大 HSV 值生成紧致阈值范围
* 不引入额外容差

**适用场景：**

* 光照稳定
* 颜色纯度高
* 需要严格控制误检的场合

---

### 2. 容差 HSV 选取（Estimating Mode）

* 在采样结果基础上自动引入容差
* 默认容差设置：

  * Hue ±10
  * Saturation ±40
  * Value ±40
* 支持实时掩膜可视化

**适用场景：**

* 光照变化明显
* 细长或发光目标
* 实际部署环境

---

### 3. 掩膜生成示例

展示如何将标定得到的 HSV 范围直接用于：

* 二值掩膜生成
* 颜色区域提取
* 中间结果保存与验证

---

## 使用说明

```python
from 1_hsv_calibration.hsv_estimate import hsv_mask_estimating_tool
lower_hsv, upper_hsv = hsv_mask_estimating_tool("img1.png")
```

---

## 设计对比

| 模式         | 是否加容差 | 特点 | 使用建议   |
| ---------- | ----- | -- | ------ |
| Exact      | 否     | 精确 | 理想实验环境 |
| Estimating | 是     | 稳定 | 实际应用环境 |

---

## 在整体系统中的位置

```
HSV 标定
   ↓
颜色分割
   ↓
几何筛选
   ↓
位姿估计 / 跟踪
```

