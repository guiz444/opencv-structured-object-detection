import cv2
import numpy as np
from datetime import datetime
from T1.HSV_estimate import hsv_mask_estimating_tool   # 导入HSV估计工具（粗略选区）
from T1.HSV_exact import hsv_mask_exact_tool           # 导入HSV精确选区工具（点击取样）

# 读取原始图像
img = cv2.imread("s.png")

# 将图像从BGR颜色空间转换为HSV空间，便于颜色分割
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 调用HSV估计工具
# 用户在弹出的窗口中选取颜色区域或点击多个点
# 函数返回两个HSV值：下界 lower、上界 upper
lower, upper = hsv_mask_estimating_tool("s.png")

# 检查是否有有效的HSV范围
if lower is None or upper is None:
    # 如果没有点击任何颜色点，无法生成掩膜
    print("未选择颜色点，无法生成掩膜")
else:
    # 将HSV上下界转换为uint8类型（OpenCV要求）
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)

    # 根据HSV范围生成掩膜
    # 掩膜中，落在颜色范围内的像素为255（白），其余为0（黑）
    mask = cv2.inRange(hsv, lower, upper)

    # 使用掩膜提取图像中对应的颜色区域
    result = cv2.bitwise_and(img, img, mask=mask)

    # 生成带时间戳的文件名，用于区分多次结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"img1_out_{timestamp}.png"

    # 保存掩膜处理后的结果图像
    cv2.imwrite(out_filename, result)
    print(f"掩膜图保存成功：{out_filename}")
