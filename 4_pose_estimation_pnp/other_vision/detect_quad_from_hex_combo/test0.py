import cv2
import numpy as np
import os
from datetime import datetime
from test import detect_black_shapes_on_yellow_with_types  # ✅ 改成新函数
"""
main_pose_estimation.py

功能：
    - 读取视频帧，检测黄色背景内黑色目标的四个角点
    - 基于角点进行PnP求解，估计物体三维姿态
    - 绘制3D坐标轴、重投影点和角点
    - 支持连续帧平滑、跳动过滤和重投影异常处理
    - 输出处理后视频及可选截图

依赖：
    - OpenCV (cv2)
    - NumPy (np)
    - test.py 中的 detect_black_shapes_on_yellow_with_types

使用：
    python main_pose_estimation.py
    - 空格：暂停/继续播放
    - S：保存当前帧
    - ESC：退出
"""

# ---------------- 相机参数 ----------------
cameraMatrix = np.array([
    [ 3500.0,    0.0,   4000.0/2 ],
    [   0.0,   3500.0,  3000.0/2 ],
    [   0.0,     0.0,      1.0    ]
], dtype=np.float32)

distCoeffs = np.array([ -0.05, 0.02, 0.0, 0.0, 0.0 ], dtype=np.float32)

# 物体 3D 坐标（左上、右上、右下、左下）
objectPoints = np.array([
    [-64, 64, 0],
    [64, 64, 0],
    [60, -60, 0],
    [-64, -64, 0],
], dtype=np.float32)

axis_length = 50
jump_threshold = 40
SMOOTHING_ALPHA = 0.6
REPROJ_REJECT_THRESHOLD = 10

prev_pts = None   # 新函数中用的prev_pts
prev_rvec = None
prev_tvec = None

# ---------------- 创建结果文件夹 ----------------
result_folder = "results"
os.makedirs(result_folder, exist_ok=True)

# ---------------- 视频输入输出 ----------------
cap = cv2.VideoCapture("stone.mp4")
if not cap.isOpened():
    print("无法打开视频")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

timestamp_video = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = os.path.join(result_folder, f"output_result_{timestamp_video}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
print(f"输出视频文件: {output_filename}")

first_frame_saved = False
paused = False

# ---------------- 主循环 ----------------
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔹 使用新版函数检测四点
        vis, quad_pts, prev_pts = detect_black_shapes_on_yellow_with_types(
            frame, prev_pts,
            params={'min_area':100, 'score_thresh':50, 'max_frame_jump':80}
        )

        if not first_frame_saved and quad_pts is not None:
            timestamp_first = datetime.now().strftime("%Y%m%d_%H%M%S")
            first_frame_name = os.path.join(result_folder, f"first_frame_detected_{timestamp_first}.png")
            cv2.imwrite(first_frame_name, vis)
            print(f"保存第一帧检测结果: {first_frame_name}")
            first_frame_saved = True

        if quad_pts is not None and len(quad_pts) == 4:
            imagePoints = np.array(quad_pts, dtype=np.float32)

            # 跳动过滤
            draw_axes = True
            if prev_pts is not None:
                dist = np.linalg.norm(imagePoints - prev_pts, axis=1).mean()
                if dist > jump_threshold:
                    draw_axes = False

            if draw_axes:
                # ---------------- 第一帧初始化 ----------------
                if prev_rvec is None or prev_tvec is None:
                    # 用 IPPE_SQUARE 求闭式解作为初值
                    retval, rvec, tvec = cv2.solvePnP(
                        objectPoints, imagePoints, cameraMatrix, distCoeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    if retval:
                        # 保证 Z>0
                        if tvec[2][0] < 0:
                            rvec = -rvec
                            tvec = -tvec
                        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()
                else:
                    # ---------------- 后续帧迭代求解 ----------------
                    retval, rvec, tvec = cv2.solvePnP(
                        objectPoints, imagePoints, cameraMatrix, distCoeffs,
                        rvec=prev_rvec,
                        tvec=prev_tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    if retval:
                        # 保证 Z>0
                        if tvec[2][0] < 0:
                            rvec = -rvec
                            tvec = -tvec

                        # 重投影
                        proj_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
                        reproj_error = np.mean(np.linalg.norm(imagePoints - proj_points.squeeze(), axis=1))

                        # 重投影过大回退上一帧
                        if reproj_error > REPROJ_REJECT_THRESHOLD:
                            rvec, tvec = prev_rvec.copy(), prev_tvec.copy()
                        else:
                            # 平滑
                            rvec = SMOOTHING_ALPHA * rvec + (1 - SMOOTHING_ALPHA) * prev_rvec
                            tvec = SMOOTHING_ALPHA * tvec + (1 - SMOOTHING_ALPHA) * prev_tvec
                            prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

                # 绘制坐标轴
                cv2.drawFrameAxes(vis, cameraMatrix, distCoeffs, rvec, tvec, axis_length)

                # 重投影点可视化
                proj_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
                for p in proj_points.squeeze().astype(int):
                    cv2.circle(vis, tuple(p), 6, (0,0,255), 2)

                # 绘制角点
                for pt in imagePoints:
                    cv2.circle(vis, tuple(pt.astype(int)), 5, (0,255,255), -1)

        # 写入输出视频
        out.write(vis)

    # 显示画面
    cv2.imshow("3D Pose Visualization", vis)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    elif key == 32:
        paused = not paused
    elif key in [ord('s'), ord('S')]:
        timestamp_img = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(result_folder, f"saved_frame_{timestamp_img}.png")
        cv2.imwrite(filename, vis)
        print(f"保存当前帧: {filename}")
