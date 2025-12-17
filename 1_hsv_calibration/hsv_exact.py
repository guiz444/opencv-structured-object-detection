import cv2
import numpy as np


def hsv_mask_exact_tool(image_path):
    """
    HSV掩膜调试工具（支持多点点击，自动计算最小/最大HSV）
    ----------------------------------------------------------
    用途：
        通过点击图像上多个像素点，自动计算出这些点的 HSV 范围，
        实时显示掩膜效果，方便手动确定颜色阈值。

    参数：
        image_path : str
            输入图像的路径

    返回：
        lower_hsv : np.array([H, S, V])
            所有点击点的 HSV 最小值（下界）
        upper_hsv : np.array([H, S, V])
            所有点击点的 HSV 最大值（上界）
    """
    # 读取图像（BGR格式）
    img_bgr = cv2.imread(image_path)

    # 转换为HSV颜色空间
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 保留一份副本，用于绘制点击点和叠加显示
    clone = img_bgr.copy()

    # 用于保存点击点的HSV值和坐标
    hsv_points = []
    point_coords = []

    # ---------------- 鼠标点击事件回调函数 ----------------
    def pick_hsv(event, x, y, flags, param):
        nonlocal hsv_points, point_coords
        # 当鼠标左键点击时
        if event == cv2.EVENT_LBUTTONDOWN:
            # 获取该像素的HSV值
            hsv_val = img_hsv[y, x]
            hsv_points.append(hsv_val)
            point_coords.append((x, y))
            print(f"Clicked HSV: {hsv_val}")  # 输出点击像素的HSV值

    # 创建窗口并绑定鼠标回调
    cv2.namedWindow("HSV Mask Tool")
    cv2.setMouseCallback("HSV Mask Tool", pick_hsv)

    # ---------------- 主循环：实时显示 ----------------
    while True:
        # 复制图像用于显示
        display_img = clone.copy()

        # 绘制所有点击过的点（红色圆点）
        for (x, y) in point_coords:
            cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

        # 如果已经点击过点，就计算HSV范围并显示掩膜
        if hsv_points:
            hsv_array = np.array(hsv_points)

            # 求所有点击点的最小HSV和最大HSV（即颜色范围）
            lower_hsv = hsv_array.min(axis=0)
            upper_hsv = hsv_array.max(axis=0)

            # 生成对应掩膜
            mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)

            # 用红色在原图上叠加显示掩膜区域
            overlay = display_img.copy()
            overlay[mask > 0] = (0, 0, 255)  # 红色覆盖
            display_img = cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0)

        # 显示调试窗口
        cv2.imshow("HSV Mask Tool", display_img)

        # 如果窗口被关闭，则退出循环
        if cv2.getWindowProperty("HSV Mask Tool", cv2.WND_PROP_VISIBLE) < 1:
            break

        # 按下 ESC 键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # 销毁所有窗口
    cv2.destroyAllWindows()

    # ---------------- 输出最终结果 ----------------
    if hsv_points:
        lower_hsv = hsv_array.min(axis=0)
        upper_hsv = hsv_array.max(axis=0)
        print(f"Final HSV range: {lower_hsv} ~ {upper_hsv}")
        return lower_hsv, upper_hsv
    else:
        print("No clicked HSV point")
        return None, None
