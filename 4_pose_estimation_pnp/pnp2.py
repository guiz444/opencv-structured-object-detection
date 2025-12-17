import cv2
import math
from imagepoints import detect_black_shapes_on_yellow
import numpy as np
import matplotlib.pyplot as plt

#### 复现，相机运动轨迹图，局部坐标系

fig = plt.figure()
# 创建3d绘图区域
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

pose_x = []
pose_y = []
pose_z = []

# ---------------- 相机参数 ----------------
cameraMatrix = np.array([
    [ 3500.0,    0.0,   4000.0/2 ],
    [   0.0,   3500.0,  3000.0/2 ],
    [   0.0,     0.0,      1.0    ]
], dtype=np.float32)

distCoeffs = np.array([ -0.05, 0.02, 0.0, 0.0, 0.0 ], dtype=np.float32)

# ---------------- 物体四角点 (mm) ----------------
objectPoints = np.array([[-75, -75, 0],
                         [75, -75, 0],
                         [75, 75, 0],
                         [-75, 75, 0]], dtype=np.float32)

axis_length = 50


def rotate_by_z(x, y, theta_z):
    outx = math.cos(theta_z) * x - math.sin(theta_z) * y
    outy = math.sin(theta_z) * x + math.cos(theta_z) * y
    return outx, outy


def rotate_by_x(y, z, theta_x):
    outy = math.cos(theta_x) * y - math.sin(theta_x) * z
    outz = math.sin(theta_x) * y + math.cos(theta_x) * z
    return outy, outz


def rotate_by_y(z, x, theta_y):
    outz = math.cos(theta_y) * z - math.sin(theta_y) * x
    outx = math.sin(theta_y) * z + math.cos(theta_y) * x
    return outz, outx


def get_euler_angle(rotation_vector):
    # 旋转顺序是z,y,x，对于相机来说就是滚转，偏航，俯仰
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    return roll, pitch, yaw


def order_points_from_squares(squares):
    """
    从检测到的四个方块中提取角点并排序
    返回顺序：左上、右上、右下、左下
    """
    if len(squares) != 4:
        return None

    # 提取四个方块的坐标
    imagePoints = np.array([
        [squares[0][0], squares[0][1]],  # 左上
        [squares[1][0] + squares[1][2], squares[1][1]],  # 右上
        [squares[2][0] + squares[2][2], squares[2][1] + squares[2][3]],  # 右下
        [squares[3][0], squares[3][1] + squares[3][3]]  # 左下
    ], dtype=np.float32)

    return imagePoints


def solve_pnp_with_squares(frame, objectPoints, cameraMatrix, distCoeffs):
    """
    使用检测到的方块进行PnP求解
    """
    # 检测黑色方块
    result_img, squares = detect_black_shapes_on_yellow(frame)

    if len(squares) == 4:
        # 提取并排序角点
        imagePoints = order_points_from_squares(squares)

        if imagePoints is not None:
            # PnP求解
            ret, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs,
                                           flags=cv2.SOLVEPNP_ITERATIVE)
            return ret, rvec, tvec, imagePoints, result_img

    return None, None, None, None, frame


# 主循环
vc = cv2.VideoCapture('stone.mp4')

while vc.isOpened():
    ret, frame = vc.read()
    if frame is None:
        break

    # 使用封装函数进行PnP求解
    ret, rvec, tvec, imagePoints, display_frame = solve_pnp_with_squares(frame, objectPoints, cameraMatrix, distCoeffs)

    if ret:
        # 提取四个角点用于显示
        A, B, C, D = imagePoints.astype(int)

        # 投影坐标轴到图像平面
        (Z_end, jacobian_z) = cv2.projectPoints(np.array([(0.0, 0.0, axis_length)]), rvec, tvec, cameraMatrix,
                                                distCoeffs)
        (X_end, jacobian_x) = cv2.projectPoints(np.array([(axis_length, 0.0, 0.0)]), rvec, tvec, cameraMatrix,
                                                distCoeffs)
        (Y_end, jacobian_y) = cv2.projectPoints(np.array([(0.0, axis_length, 0.0)]), rvec, tvec, cameraMatrix,
                                                distCoeffs)
        (O, jacobian_o) = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rvec, tvec, cameraMatrix, distCoeffs)

        Z_end = (int(Z_end[0][0][0]), int(Z_end[0][0][1]))
        X_end = (int(X_end[0][0][0]), int(X_end[0][0][1]))
        Y_end = (int(Y_end[0][0][0]), int(Y_end[0][0][1]))
        O_ = (int(O[0][0][0]), int(O[0][0][1]))

        # 在图像上绘制坐标轴
        cv2.line(display_frame, O_, X_end, (50, 255, 50), 10)
        cv2.line(display_frame, O_, Y_end, (50, 50, 255), 10)
        cv2.line(display_frame, O_, Z_end, (255, 50, 50), 10)
        cv2.putText(display_frame, "X", X_end, cv2.FONT_HERSHEY_SIMPLEX, 3, (5, 150, 5), 10)
        cv2.putText(display_frame, "Y", Y_end, cv2.FONT_HERSHEY_SIMPLEX, 3, (5, 5, 150), 10)
        cv2.putText(display_frame, "Z", Z_end, cv2.FONT_HERSHEY_SIMPLEX, 3, (150, 5, 5), 10)

        # 标记角点
        cv2.circle(display_frame, tuple(A), 3, (255, 255, 255), 10)
        cv2.circle(display_frame, tuple(B), 3, (255, 255, 255), 10)
        cv2.circle(display_frame, tuple(C), 3, (255, 255, 255), 10)
        cv2.circle(display_frame, tuple(D), 3, (255, 255, 255), 10)

        cv2.putText(display_frame, "A", tuple(A + [15, 15]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.putText(display_frame, "B", tuple(B + [15, 15]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.putText(display_frame, "C", tuple(C + [15, 15]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        cv2.putText(display_frame, "D", tuple(D + [15, 15]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)

        # 显示位姿信息
        xt, yt, zt = tvec.flatten()
        xr, yr, zr = rvec.flatten()

        rvec_str = f'rotation_vector: ({xr:.2f}, {yr:.2f}, {zr:.2f})'
        tvec_str = f'translation_vector: ({xt:.2f}, {yt:.2f}, {zt:.2f})'

        cv2.putText(display_frame, tvec_str, [30, 30], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(display_frame, rvec_str, [30, 80], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # 计算欧拉角
        roll, pitch, yaw = get_euler_angle(rvec)

        # 坐标转换
        Xc, Yc, Zc = tvec.flatten()
        Xc, Yc = rotate_by_z(Xc, Yc, -roll)
        Zc, Xc = rotate_by_y(Zc, Xc, -yaw)
        Yc, Zc = rotate_by_x(Yc, Zc, -pitch)

        # 记录轨迹
        pose_x.append(-Xc)
        pose_y.append(-Yc)
        pose_z.append(-Zc)

        position_str = f'position: ({-Xc:.2f}, {-Yc:.2f}, {-Zc:.2f})'
        cv2.putText(display_frame, position_str, [30, 130], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # 显示结果
    cv2.imshow("res", display_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

vc.release()
cv2.destroyAllWindows()

# 绘制3D轨迹
pose_x = np.array(pose_x)
pose_y = np.array(pose_y)
pose_z = np.array(pose_z)

ax.plot3D(pose_x, pose_y, pose_z, linewidth=0.4)
ax.set_title('Trace')
plt.show()