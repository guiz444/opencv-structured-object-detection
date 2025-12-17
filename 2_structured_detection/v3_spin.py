import cv2
import numpy as np
from datetime import datetime

## cv2.minAreaRect(cnt) → 最小旋转矩形  cv2.boxPoints(rect) → 计算旋转矩形四点
def detect_blue_light_arrow(img_path):

    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 30, 180], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    result_img = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        # 绘制原始轮廓（绿色）
        cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)

        # 使用最小外接矩形（可旋转矩形），比普通 boundingRect 更精确
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect  # 获取矩形中心坐标、宽高和旋转角度

        if w < 5 or h < 5:
            continue

        # 获取矩形四个顶点坐标，并转换为整数
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)

        # 绘制红色旋转矩形
        cv2.drawContours(result_img, [box], 0, (0, 0, 255), 2)

        # 保存中心点、宽高和旋转角度（新增 angle 参数）
        results.append(((int(x), int(y)), (int(w), int(h)), angle))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] 检测到 {len(results)} 个箭头灯条区域")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"arrow_detect_v3_{timestamp}.png"
    cv2.imwrite(save_name, result_img)
    print(f"已保存检测结果：{save_name}")

    return mask, result_img, results


if __name__ == "__main__":
    mask, result_img, results = detect_blue_light_arrow("img2.png")
