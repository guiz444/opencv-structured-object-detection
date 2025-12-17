import cv2
import numpy as np
import os
from datetime import datetime
from imagepoints import detect_black_shapes_on_yellow

##固定世界坐标系建系（物理上始终正交，不随pnp变化，投影到图像上受PnP影响）

# ---------------- 相机参数 ----------------
cameraMatrix = np.array([
    [ 3500.0,    0.0,   4000.0/2 ],
    [   0.0,   3500.0,  3000.0/2 ],
    [   0.0,     0.0,      1.0    ]
], dtype=np.float32)

distCoeffs = np.array([ -0.05, 0.02, 0.0, 0.0, 0.0 ], dtype=np.float32)

objectPoints = np.array([[-75, 75, 0],
                         [75, 75, 0],
                         [75, -75, 0],
                         [-75, -75, 0]], dtype=np.float32)

axis_length = 50          # 坐标轴长度
jump_threshold = 40       # 跳动阈值
prev_imagePoints = None   # 上一帧角点

# ---------------- 创建结果文件夹 ----------------
result_folder = "results"
os.makedirs(result_folder, exist_ok=True)

# ---------------- 视频输入输出 ----------------
cap = cv2.VideoCapture("3.mp4")
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

        # 检测黄色背景上的黑色方块
        result_img, squares = detect_black_shapes_on_yellow(frame)

        # 保存第一帧
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
                [squares[1][0] + squares[1][2], squares[1][1]],
                [squares[2][0] + squares[2][2], squares[2][1] + squares[2][3]],
                [squares[3][0], squares[3][1] + squares[3][3]]
            ], dtype=np.float32)

            # 跳动过滤
            if prev_imagePoints is None:
                draw_axes = True
            else:
                dist = np.linalg.norm(imagePoints - prev_imagePoints, axis=1).mean()
                if dist < jump_threshold:
                    draw_axes = True
            prev_imagePoints = imagePoints.copy()

            # PnP求解
            if draw_axes:
                retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs)
                if retval:
                    # ---------------- 世界坐标系绘制坐标轴 ----------------
                    origin_obj = np.mean(objectPoints, axis=0).reshape(3, 1)

                    vec_x_obj = np.array([axis_length, 0, 0], dtype=np.float32)
                    vec_y_obj = np.array([0, axis_length, 0], dtype=np.float32)
                    vec_z_obj = np.array([0, 0, axis_length], dtype=np.float32)

                    pts_3d = np.array([
                        origin_obj.squeeze() + vec_x_obj,
                        origin_obj.squeeze() + vec_y_obj,
                        origin_obj.squeeze() + vec_z_obj
                    ], dtype=np.float32).reshape(-1, 3)

                    proj_pts, _ = cv2.projectPoints(pts_3d, rvec, tvec, cameraMatrix, distCoeffs)
                    origin_2d, _ = cv2.projectPoints(origin_obj, rvec, tvec, cameraMatrix, distCoeffs)
                    origin_2d = tuple(origin_2d.squeeze().astype(int))
                    proj_pts = proj_pts.squeeze().astype(int)

                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X红, Y绿, Z蓝
                    for i in range(3):
                        cv2.line(frame, origin_2d, tuple(proj_pts[i]), colors[i], 3)

                    # ---------------- 坐标轴正交性验证 ----------------
                    dot_xy = np.dot(vec_x_obj, vec_y_obj)
                    dot_xz = np.dot(vec_x_obj, vec_z_obj)
                    dot_yz = np.dot(vec_y_obj, vec_z_obj)
                    len_x = np.linalg.norm(vec_x_obj)
                    len_y = np.linalg.norm(vec_y_obj)
                    len_z = np.linalg.norm(vec_z_obj)

                    # ---------------- 重投影误差 ----------------
                    proj_points, _ = cv2.projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs)
                    reproj_error = np.mean(np.linalg.norm(imagePoints - proj_points.squeeze(), axis=1))

                    # ---------------- 旋转矩阵正交性检查 ----------------
                    R, _ = cv2.Rodrigues(rvec)
                    dot01 = np.dot(R[:, 0], R[:, 1])
                    dot02 = np.dot(R[:, 0], R[:, 2])
                    dot12 = np.dot(R[:, 1], R[:, 2])
                    detR = np.linalg.det(R)

                    # ---------------- 欧拉角 ----------------
                    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
                    if sy > 1e-6:
                        yaw = np.arctan2(R[1, 0], R[0, 0])
                        pitch = np.arctan2(-R[2, 0], sy)
                        roll = np.arctan2(R[2, 1], R[2, 2])
                    else:
                        yaw = np.arctan2(-R[1, 2], R[1, 1])
                        pitch = np.arctan2(-R[2, 0], sy)
                        roll = 0

                    # ---------------- 控制台输出 ----------------
                    print(f"\n=== 帧检测结果 ===")
                    print(f"坐标轴正交性检查:")
                    print(f"dot(X,Y)={dot_xy:.4f}, dot(X,Z)={dot_xz:.4f}, dot(Y,Z)={dot_yz:.4f}")
                    print(f"长度: |X|={len_x:.2f}, |Y|={len_y:.2f}, |Z|={len_z:.2f}")
                    print(f"重投影误差: {reproj_error:.2f} 像素")
                    print(f"正交性: dot01={dot01:.4f}, dot02={dot02:.4f}, dot12={dot12:.4f}, det={detR:.4f}")
                    print("Yaw/Pitch/Roll (deg):", np.degrees([yaw, pitch, roll]))
                    print(f"Z 距离: {tvec[2][0]:.1f} mm")

                    # ---------------- 图像叠加 ----------------
                    cv2.putText(frame, f"Err: {reproj_error:.2f}px", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(frame, f"Z: {tvec[2][0]:.1f}mm", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    cv2.putText(frame, f"Yaw:{np.degrees(yaw):.1f} Pitch:{np.degrees(pitch):.1f} Roll:{np.degrees(roll):.1f}",
                                (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

                    # ---------------- 绘制重投影点 ----------------
                    for p in proj_points.squeeze().astype(int):
                        cv2.circle(frame, tuple(p), 6, (0,0,255), 2)

            # 绘制检测角点
            for pt in imagePoints:
                cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 255), -1)

        # 写入输出视频
        out.write(frame)

    # 显示画面
    cv2.imshow("3D Pose Visualization", frame)
    key = cv2.waitKey(30) & 0xFF
    if key == 27:  # ESC退出
        break
    elif key == 32:  # 空格暂停/继续
        paused = not paused
    elif key == ord('s') or key == ord('S'):  # 保存当前帧
        timestamp_img = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(result_folder, f"saved_frame_{timestamp_img}.png")
        cv2.imwrite(filename, frame)
        print(f"保存当前帧: {filename}")

# ---------------- 释放资源 ----------------
cap.release()
out.release()
cv2.destroyAllWindows()
