"""
===============================================================================
3D Pose Estimation and Visualization – 优化版
===============================================================================

文件说明：
本模块用于对视频中黄色背景上的黑色矩形进行检测，并基于四角点
计算物体的 3D 姿态（旋转和平移），同时绘制三维坐标轴和重投影点。

主要功能：
1. 自动检测黑色矩形，提取四角点。
2. 根据物体坐标系和像素坐标求解 PnP 问题，得到 rvec/tvec。
3. 自动绘制坐标轴 (X:红, Y:绿, Z:蓝)。
4. 跳动过滤、姿态平滑和平滑异常剔除，提高视频连续帧的稳定性。
5. 支持暂停、单帧保存和视频输出。

物体坐标系约定：
- 原点：矩形中心
- X 轴：从左到右
- Y 轴：从上到下
- Z 轴：垂直矩形平面，指向摄像机前方
- 角点顺序：左上 → 右上 → 右下 → 左下
- 矩形大小单位 mm，四边长 150x150 mm

与第一版对比：
--------------------------------------
1. 姿态初始化：
   - 第一版：每帧独立求解 PnP，无连续帧参考，容易出现跳动。
   - 优化版：上一帧 rvec/tvec 用作初始值 (useExtrinsicGuess)，并加平滑。

2. 跳动过滤：
   - 第一版：无角点跳动检查。
   - 优化版：加入 `jump_threshold`，角点移动过大时跳过更新。

3. 姿态平滑：
   - 第一版：无平滑处理。
   - 优化版：使用 `SMOOTHING_ALPHA` 对 rvec/tvec 加权融合，稳定姿态曲线。

4. 异常剔除：
   - 第一版：直接使用当前求解结果。
   - 优化版：根据重投影误差 `REPROJ_REJECT_THRESHOLD` 判定异常，使用上一帧姿态替代。

5. 绘图：
   - 第一版：简单绘制坐标轴和角点。
   - 优化版：绘制坐标轴 + 重投影点，并显示欧拉角、Z 距离和重投影误差。

6. 视频与图像管理：
   - 第一版：无第一帧保存。
   - 优化版：首次检测到矩形时保存第一帧，并支持按键保存任意帧。
7. 角点顺序与坐标系对比说明：
  第一版：左上 → 左下 → 右上 → 右下
  优化版：
     提取顺序（图像坐标系 y向下, x向右）：
     左上 → 右上 → 右下 → 左下
     对应 objectPoints：
     [-75,  75, 0]   左上
     [ 75,  75, 0]   右上
     [ 75, -75, 0]   右下
     [-75, -75, 0]   左下


总结：
优化版在连续视频帧处理时具有更高稳定性、更少跳动，并可提供姿态质量
评价信息（重投影误差、正交性、欧拉角），适合精细化视频分析和可视化。

===============================================================================
"""

import cv2
import numpy as np
import os
from datetime import datetime
from imagepoints import detect_black_shapes_on_yellow

# ---------------- 参数配置 ----------------
CAMERA_MATRIX = np.array([
    [3500.0, 0.0, 4000.0/2],
    [0.0, 3500.0, 3000.0/2],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

DIST_COEFFS = np.array([-0.05, 0.02, 0.0, 0.0, 0.0], dtype=np.float32)
OBJECT_POINTS = np.array([
    [-75, 75, 0],  # 左上角
    [75, 75, 0],  # 右上角
    [75, -75, 0],  # 右下角
    [-75, -75, 0],  # 左下角
], dtype=np.float32)

AXIS_LENGTH = 50
JUMP_THRESHOLD = 40
SMOOTHING_ALPHA = 0.6
REPROJ_REJECT_THRESHOLD = 10

RESULT_FOLDER = "results"
os.makedirs(RESULT_FOLDER, exist_ok=True)

VIDEO_PATH = "stone.mp4"

# ---------------- 状态变量 ----------------
prev_imagePoints = None
prev_rvec = None
prev_tvec = None
first_frame_saved = False
paused = False

# ---------------- 工具函数 ----------------
def compute_distance(pts1, pts2):
    return np.linalg.norm(pts1 - pts2, axis=1).mean()

def solve_pnp_with_guess(obj_pts, img_pts, rvec_prev=None, tvec_prev=None):
    if rvec_prev is not None and tvec_prev is not None:
        return cv2.solvePnP(
            obj_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS,
            rvec=rvec_prev, tvec=tvec_prev,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
    else:
        return cv2.solvePnP(obj_pts, img_pts, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_IPPE_SQUARE)

def smooth_pose(rvec, tvec, rvec_prev, tvec_prev):
    rvec_s = SMOOTHING_ALPHA * rvec + (1 - SMOOTHING_ALPHA) * rvec_prev
    tvec_s = SMOOTHING_ALPHA * tvec + (1 - SMOOTHING_ALPHA) * tvec_prev
    return rvec_s, tvec_s

def compute_euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        yaw = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        yaw = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        roll = 0
    return yaw, pitch, roll

# ---------------- 视频处理 ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("无法打开视频")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

timestamp_video = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = os.path.join(RESULT_FOLDER, f"output_result_{timestamp_video}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
print(f"输出视频文件: {output_filename}")

# ---------------- 主循环 ----------------
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        result_img, squares = detect_black_shapes_on_yellow(frame)

        # 保存第一帧检测结果
        if not first_frame_saved and squares:
            timestamp_first = datetime.now().strftime("%Y%m%d_%H%M%S")
            first_frame_name = os.path.join(RESULT_FOLDER, f"first_frame_detected_{timestamp_first}.png")
            cv2.imwrite(first_frame_name, result_img)
            print(f"保存第一帧检测结果: {first_frame_name}")
            first_frame_saved = True

        # 只处理 4 个角点的情况
        if len(squares) == 4:
            imagePoints = np.array([
                [squares[0][0], squares[0][1]],
                [squares[1][0]+squares[1][2], squares[1][1]],
                [squares[2][0]+squares[2][2], squares[2][1]+squares[2][3]],
                [squares[3][0], squares[3][1]+squares[3][3]]
            ], dtype=np.float32)

            draw_axes = False
            if prev_imagePoints is None or compute_distance(imagePoints, prev_imagePoints) < JUMP_THRESHOLD:
                draw_axes = True

            prev_imagePoints = imagePoints.copy()

            if draw_axes:
                retval, rvec, tvec = solve_pnp_with_guess(OBJECT_POINTS, imagePoints, prev_rvec, prev_tvec)

                if retval:
                    proj_points, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
                    reproj_error = np.mean(np.linalg.norm(imagePoints - proj_points.squeeze(), axis=1))

                    # 重投影误差过大，使用上一帧
                    if prev_rvec is not None and prev_tvec is not None and reproj_error > REPROJ_REJECT_THRESHOLD:
                        rvec, tvec = prev_rvec.copy(), prev_tvec.copy()
                    elif prev_rvec is not None and prev_tvec is not None:
                        rvec, tvec = smooth_pose(rvec, tvec, prev_rvec, prev_tvec)

                    prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

                    # 绘制坐标轴
                    cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, AXIS_LENGTH)

                    # 绘制重投影点
                    for p in proj_points.squeeze().astype(int):
                        cv2.circle(frame, tuple(p), 6, (0,0,255), 2)

                    # 欧拉角 + 正交性
                    R, _ = cv2.Rodrigues(rvec)
                    dot01 = np.dot(R[:,0], R[:,1])
                    dot02 = np.dot(R[:,0], R[:,2])
                    dot12 = np.dot(R[:,1], R[:,2])
                    detR = np.linalg.det(R)
                    yaw, pitch, roll = compute_euler(R)

                    # 输出信息
                    print(f"\n=== 帧检测结果 ===")
                    print(f"重投影误差: {reproj_error:.2f} px")
                    print(f"正交性: dot01={dot01:.4f}, dot02={dot02:.4f}, dot12={dot12:.4f}, det={detR:.4f}")
                    print(f"Yaw/Pitch/Roll (deg): {np.degrees([yaw,pitch,roll])}")
                    print(f"Z 距离: {tvec[2][0]:.1f} mm")

                    # 屏幕显示
                    cv2.putText(frame, f"Err:{reproj_error:.2f}px", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
                    cv2.putText(frame, f"Z:{tvec[2][0]:.1f}mm", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
                    cv2.putText(frame, f"Yaw:{np.degrees(yaw):.1f} Pitch:{np.degrees(pitch):.1f} Roll:{np.degrees(roll):.1f}",
                                (30,130), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,0),2)

            # 绘制角点
            for pt in imagePoints:
                cv2.circle(frame, tuple(pt.astype(int)),5,(0,255,255),-1)

        out.write(frame)

    # 显示画面
    cv2.imshow("3D Pose Visualization", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC 退出
        break
    elif key == 32:  # 空格暂停
        paused = not paused
    elif key in [ord('s'), ord('S')]:  # 保存当前帧
        timestamp_img = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RESULT_FOLDER, f"saved_frame_{timestamp_img}.png")
        cv2.imwrite(filename, frame)
        print(f"保存当前帧: {filename}")

cap.release()
out.release()
cv2.destroyAllWindows()
