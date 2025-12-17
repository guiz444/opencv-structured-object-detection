import cv2
import numpy as np


def hsv_mask_estimating_tool(image_path):
    """
    HSV掩膜调试工具（支持多点点击，自动计算最小/最大HSV）

    功能：
    - 鼠标左键点击图像中的多个区域，自动统计这些像素点的HSV范围；
    - 自动计算出最小和最大HSV值；
    - 实时在图像上显示掩膜效果；
    - 按下 Esc 或关闭窗口即可退出。

    参数：
    - image_path: 图像路径

    返回：
    - lower_hsv: np.array([H_min, S_min, V_min])
    - upper_hsv: np.array([H_max, S_max, V_max])
    """

    # 读取图像（BGR 格式）
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("无法读取图像，请检查路径是否正确。")
        return None, None

    # 转换为 HSV 颜色空间
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    clone = img_bgr.copy()  # 用于显示点击标记

    # 存储点击的 HSV 值与对应坐标
    hsv_points = []
    point_coords = []

    # 鼠标点击事件回调函数
    def pick_hsv(event, x, y, flags, param):
        nonlocal hsv_points, point_coords
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            hsv_val = img_hsv[y, x]  # 取点击点的 HSV 值
            hsv_points.append(hsv_val)
            point_coords.append((x, y))
            print(f"Clicked HSV: {hsv_val}")

    # 创建窗口并绑定鼠标回调
    cv2.namedWindow("HSV Mask Tool")
    cv2.setMouseCallback("HSV Mask Tool", pick_hsv)

    # 主循环：实时显示
    while True:
        display_img = clone.copy()

        # 在已点击位置画红点
        for (x, y) in point_coords:
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

        # 若已有点击点，则计算 HSV 范围并生成掩膜
        if hsv_points:
            hsv_array = np.array(hsv_points)

            # 当前点击点的最小/最大HSV
            lower_hsv = hsv_array.min(axis=0)
            upper_hsv = hsv_array.max(axis=0)

            # 增加一定的容差，使范围更宽!!!!!!!
            tol = np.array([5, 20, 20])  # H,S,V 容差
            lower_hsv = np.clip(lower_hsv - tol, 0, [179, 255, 255])
            upper_hsv = np.clip(upper_hsv + tol, 0, [179, 255, 255])

            # 根据范围生成掩膜
            mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

            # 用红色覆盖掩膜区域并融合显示
            overlay = display_img.copy()
            overlay[mask > 0] = (0, 0, 255)
            display_img = cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0)

        # 显示结果图
        cv2.imshow("HSV Mask Tool", display_img)

        # 若窗口被关闭或按下 Esc 键则退出
        if cv2.getWindowProperty("HSV Mask Tool", cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键退出
            break

    # 销毁窗口
    cv2.destroyAllWindows()

    # 最终输出点击范围的HSV上下限
    if hsv_points:
        hsv_array = np.array(hsv_points)
        lower_hsv = hsv_array.min(axis=0)
        upper_hsv = hsv_array.max(axis=0)

        # 最终再加大一点容差，适合用于真实检测
        tol = np.array([10, 40, 40])  # H,S,V 容差
        lower_hsv = np.clip(lower_hsv - tol, 0, [179, 255, 255])
        upper_hsv = np.clip(upper_hsv + tol, 0, [179, 255, 255])

        print(f"Final HSV range: {lower_hsv} ~ {upper_hsv}")
        return lower_hsv, upper_hsv
    else:
        print("No clicked HSV points")
        return None, None
