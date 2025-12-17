import cv2
import numpy as np
from datetime import datetime

## cv2.boundingRect(cnt) → 普通矩形(水平)   绘制矩形 cv2.rectangle()
def detect_blue_light_arrow_bounding(img_path):
    """
    检测蓝色箭头灯条区域，并绘制轮廓与外接矩形

    参数：
        img_path : str
            输入图像路径

    返回：
        mask : 蓝色区域的二值掩膜
        result_img : 绘制检测结果的图像
        results : 检测到的矩形区域列表 [(左上角坐标), (宽, 高)]
    """

    # 读取输入图像
    img = cv2.imread(img_path)

    # 转换为HSV颜色空间，便于进行颜色分割
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义蓝色的HSV阈值范围
    lower_blue = np.array([90, 30, 180], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)

    # 根据HSV范围生成掩膜
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 使用形态学闭运算去除噪点并连接相近区域
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找掩膜中的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 保存检测结果的列表和可视化图像
    results = []
    result_img = img.copy()

    # 遍历每个轮廓
    for cnt in contours:
        # 计算轮廓面积，过滤掉太小的噪点
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        # 绘制轮廓（绿色）
        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)

        # 获取该轮廓的外接矩形
        x, y, w, h = cv2.boundingRect(cnt)

        # 过滤掉过小的矩形区域
        if w < 5 or h < 5:
            continue

        # 绘制矩形框（红色）
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 保存检测结果
        results.append(((x, y), (w, h)))

    # 输出检测信息
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 检测到 {len(results)} 个箭头灯条区域")

    # 生成时间戳文件名，保存检测结果图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"arrow_detect_v1_{timestamp}.png"
    cv2.imwrite(save_name, result_img)
    print(f"已保存检测结果：{save_name}")

    # 返回掩膜、结果图和检测到的区域信息
    return mask, result_img, results


if __name__ == "__main__":
    # 测试主函数，输入图像路径为 img2.png
    mask, result_img, results = detect_blue_light_arrow_bounding("img2.png")
