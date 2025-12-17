import cv2
import numpy as np
import os
from datetime import datetime
from imagepoints import detect_black_shapes_on_yellow

## 自动绘图+保留上一帧的姿态，用于初始化下一帧，并进行平滑

# ---------------- 相机参数 ----------------
cameraMatrix = np.array([
    [ 3500.0,    0.0,   4000.0/2 ],
    [   0.0,   3500.0,  3000.0/2 ],
    [   0.0,     0.0,      1.0    ]
], dtype=np.float32)

distCoeffs = np.array([ -0.05, 0.02, 0.0, 0.0, 0.0 ], dtype=np.float32)


# 物体 3D 坐标（左上、右上、右下、左下）
objectPoints = np.array([
    [-75, 75, 0],
    [75, 75, 0],
    [75, -75, 0],
    [-75, -75, 0],
], dtype=np.float32)

axis_length = 50          # 坐标轴长度
jump_threshold = 40       # 跳动阈值
SMOOTHING_ALPHA = 0.6
REPROJ_REJECT_THRESHOLD = 10

prev_imagePoints = None
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

        result_img, squares = detect_black_shapes_on_yellow(frame)

        if not first_frame_saved and squares:
            timestamp_first = datetime.now().strftime("%Y%m%d_%H%M%S")
            first_frame_name = os.path.join(result_folder, f"first_frame_detected_{timestamp_first}.png")
            cv2.imwrite(first_frame_name, result_img)
            print(f"保存第一帧检测结果: {first_frame_name}")
            first_frame_saved = True

        draw_axes = False
        if len(squares) == 4:
            # 提取角点
            imagePoints = np.array([
                [squares[0][0], squares[0][1]],
                [squares[1][0]+squares[1][2], squares[1][1]],
                [squares[2][0]+squares[2][2], squares[2][1]+squares[2][3]],
                [squares[3][0], squares[3][1]+squares[3][3]]
            ], dtype=np.float32)

            # 跳动过滤
            if prev_imagePoints is None:
                draw_axes = True
            else:
                dist = np.linalg.norm(imagePoints - prev_imagePoints, axis=1).mean()
                if dist < jump_threshold:
                    draw_axes = True

            prev_imagePoints = imagePoints.copy()

            # ---------------- SolvePnP ----------------
            if draw_axes:
                if prev_rvec is not None and prev_tvec is not None:
                    retval, rvec, tvec = cv2.solvePnP(
                        objectPoints, imagePoints, cameraMatrix, distCoeffs,
                        rvec=prev_rvec, tvec=prev_tvec,
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                else:
                    retval, rvec, tvec = cv2.solvePnP(
                        objectPoints, imagePoints, cameraMatrix, distCoeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                if retval:
                    # 重投影误差
                    proj_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
                    reproj_error = np.mean(np.linalg.norm(imagePoints - proj_points.squeeze(), axis=1))

                    # 如果误差过大，使用上一帧
                    if prev_rvec is not None and prev_tvec is not None and reproj_error > REPROJ_REJECT_THRESHOLD:
                        rvec, tvec = prev_rvec.copy(), prev_tvec.copy()
                    else:
                        # 姿态平滑
                        if prev_rvec is not None and prev_tvec is not None:
                            rvec = SMOOTHING_ALPHA * rvec + (1 - SMOOTHING_ALPHA) * prev_rvec
                            tvec = SMOOTHING_ALPHA * tvec + (1 - SMOOTHING_ALPHA) * prev_tvec

                        prev_rvec, prev_tvec = rvec.copy(), tvec.copy()

                    # ---------------- 自动绘制坐标轴 ----------------
                    cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, axis_length)

                    # 绘制重投影点
                    for p in proj_points.squeeze().astype(int):
                        cv2.circle(frame, tuple(p), 6, (0,0,255), 2)

                    # ---------------- 欧拉角 + 正交性 ----------------
                    R, _ = cv2.Rodrigues(rvec)
                    dot01 = np.dot(R[:,0], R[:,1])
                    dot02 = np.dot(R[:,0], R[:,2])
                    dot12 = np.dot(R[:,1], R[:,2])
                    detR = np.linalg.det(R)
                    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
                    if sy > 1e-6:
                        yaw = np.arctan2(R[1,0], R[0,0])
                        pitch = np.arctan2(-R[2,0], sy)
                        roll = np.arctan2(R[2,1], R[2,2])
                    else:
                        yaw = np.arctan2(-R[1,2], R[1,1])
                        pitch = np.arctan2(-R[2,0], sy)
                        roll = 0

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

        # 写入输出视频
        out.write(frame)

    # 显示画面
    cv2.imshow("3D Pose Visualization", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    elif key == 32:
        paused = not paused
    elif key in [ord('s'), ord('S')]:
        timestamp_img = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(result_folder, f"saved_frame_{timestamp_img}.png")
        cv2.imwrite(filename, frame)
        print(f"保存当前帧: {filename}")

cap.release()
out.release()
cv2.destroyAllWindows()
