import cv2
import numpy as np
"""
===========================================================
仅实时检测版本，无位姿解算，但对物体不同视角下的识别适应性增强
===========================================================

模块功能：
-------------
本脚本用于基于摄像头或视频帧实时检测金矿石或银矿石的标记（箭头和方块）。
核心功能包括：
1. 视频或摄像头输入，逐帧获取图像；
2. 对图像中的黄色或银色区域进行分割，提取目标矿石区域；
3. 对 ROI 区域进行轮廓检测与多边形近似，识别箭头和方块标记；
4. 根据标记组合判断矿石发现状态（如 "Gold stone discovery" 或 "detecting"）；
5. 在图像上叠加可视化结果，包括轮廓、高亮和状态信息；
6. 支持按键保存当前帧图像。

"""

# 初始化标记个数：
left_arrow_sign = 0
left_square_sign = 0
right_arrow_sign = 0
right_square_sign = 0
rd_arrow_sign = 0
rd_square_sign = 0
ld_arrow_sign = 0
ld_square_sign = 0
posture = 'normal'
ip_address = '10.13.81.95'  # 使用droidcam调用手机摄像头，应自行更改ip地址
port = 4747  # 端口
# capture = cv2.VideoCapture(f'http://{ip_address}:{port}/mjpegfeed')  # 使用网站调用摄像头
capture = cv2.VideoCapture(0)
# 定义黄色范围：
lower_yellow = np.array([15, 100, 100])
upper_yellow = np.array([35, 255, 255])
silver_lower = np.array([0, 0, 50])
silver_upper = np.array([180, 255, 255])


# 获取输入：
user_input = input("键入a检测金矿石，键入b检测银矿石：")
# 检测模式选择：
while True:
    ret, image = capture.read()
    if user_input == "a":
        # 提取画面中的黄色部分，找到金矿石的大致位置：
        resized_img = cv2.resize(image, (500, 500))
        hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)  # 创建掩模
        yellow_img = cv2.bitwise_and(resized_img, resized_img, mask=mask)
        yellow_img[np.where(mask == 0)] = [0, 0, 0]  # 将非黄色区域设置为黑色
        gray = cv2.cvtColor(yellow_img, cv2.COLOR_BGR2GRAY)
        s, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 当目标不在视野内时，保活程序
        if len(contours) == 0:
            black_img = np.zeros((500, 500, 3), np.uint8)
            resized_img = black_img
            r1 = black_img
            resized_roi = black_img
            result_image = black_img
        else:
            max_contour = max(contours, key=cv2.contourArea)
            # 计算最小外接矩形,判断画面中是否有金矿石
            rect = cv2.minAreaRect(max_contour)
            w1, h1 = rect[1]
            area = w1*h1
            if area < 1000:
                print("未检测到金矿石！")
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # 获取ROI
            x, y, w, h = cv2.boundingRect(box)
            rotated_roi = resized_img[y:y + h, x:x + w]

            # 调整ROI的方向
            if rotated_roi.size != 0:
                # 放大ROI
                scale_factor = 2
                scaled_roi = cv2.resize(rotated_roi, None, fx=scale_factor, fy=scale_factor,
                                        interpolation=cv2.INTER_LINEAR)
                resized_roi = cv2.resize(scaled_roi, (500, 500))
            else:
                width, height = 500, 500
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                blank[1, 1] = (255, 255, 255)
                resized_roi = blank

            # 找到左上角标记，并使用形状等计算几何判断
            # 对ROI区进行轮廓处理，形状判断
            ROI = resized_roi
            gray2 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            _, binary2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)

            r1 = cv2.Canny(binary2, 100, 255)
            contours2, _ = cv2.findContours(r1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 初始化一个和原图像大小相同的空白图像，用于绘制符合条件的轮廓
            highlighted_image = np.zeros_like(resized_roi)

            # 遍历所有轮廓
            for contour in contours2:
                # 计算轮廓的面积
                area = cv2.contourArea(contour)

                # 检查面积是否在指定范围内,这个面积范围包括了箭头形标记以及方块标记
                if 1000 <= area <= 8000:
                    # 计算轮廓的边界框
                    x, y, w, h = cv2.boundingRect(contour)
                    # 检查边界框是否位于图像的左上角区域
                    # 这里假设左上角区域是图像左上角的1/4部分
                    if x <= resized_roi.shape[1] // 2 and y <= resized_roi.shape[0] // 2:
                        # 计算轮廓的近似多边形
                        epsilon = 0.02 * cv2.arcLength(contour, True)  # 计算轮廓的周长，然后乘以0.02作为精度参数
                        approx = cv2.approxPolyDP(contour, epsilon, True)  # 计算轮廓的近似多边形，True”表示多边形是闭合的
                        # 获取包含轮廓的最小矩形的长宽：
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        # 判断是否为六边形：若是，则初步筛选为箭头型标记
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            # 绘制轮廓
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            print('left_arrow_sign get')
                            left_arrow_sign = 1
                            # 在空白图像上绘制轮廓
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                            # 绘制轮廓
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            print('left_square_sign get')
                            left_square_sign = 1
                            # 在空白图像上绘制轮廓
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    # 检测右上角四分之一的区域：具体检测方式与上方相同
                    if x >= resized_roi.shape[1] // 2 and y <= resized_roi.shape[0] // 2:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            right_arrow_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            right_square_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    # 检测左下角标记：
                    if x <= resized_roi.shape[1] // 2 and y >= resized_roi.shape[0] // 2:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            ld_arrow_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            ld_square_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    # 检测右下角：
                    if x >= resized_roi.shape[1] // 2 and y >= resized_roi.shape[0] // 2:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            rd_arrow_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.75 <= aspect_ratio <= 1.35:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            rd_square_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
            # 将绘制了轮廓的图像与原图像合并，以突出显示轮廓
            result_image = cv2.addWeighted(resized_roi, 0.4, highlighted_image, 0.6, 0)
            # 数据可视化：
            cv2.putText(result_image, 'la:' + str(left_arrow_sign), (3, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'ls:' + str(left_square_sign), (3, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'ra:' + str(right_arrow_sign), (3, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'rs:' + str(right_square_sign), (3, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'lds:' + str(ld_square_sign), (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'lda:' + str(ld_arrow_sign), (40, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'rds:' + str(rd_square_sign), (40, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'rda:' + str(rd_arrow_sign), (40, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'posture:' + str(posture), (3, 120), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (255, 0, 0), 1)
            # 满足金矿石的条件应该为检测到至少两个个标记：
            if (left_square_sign == 0 and left_arrow_sign == 0) or (right_arrow_sign == 0 and right_square_sign == 0):
                cv2.putText(result_image, 'detecting', (3, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 1)
            if (left_square_sign == 1 or left_arrow_sign == 1) and (right_arrow_sign == 1 or right_square_sign == 1):
                cv2.putText(result_image, 'Gold stone discovery', (3, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
                            (0, 255, 0), 1)
                if left_square_sign == 1 and rd_square_sign == 1:
                    left_square_sign = 0
                    rd_square_sign = 0
                    posture = 'rotated'
                if left_arrow_sign == 1 and rd_arrow_sign == 1:
                    left_arrow_sign = 0
                    rd_arrow_sign = 0
                    posture = 'rotated'
                if left_arrow_sign == 1 and rd_square_sign == 1:
                    left_arrow_sign = 0
                    rd_square_sign = 0
                    posture = 'normal'
        cv2.drawContours(resized_img, contours, -1, (0, 0, 255), 2)
        cv2.imshow('yellow img:', yellow_img)  # 不作处理，直接输出
        cv2.imshow('img', resized_img)  # 经处理
        cv2.imshow('r1', r1)  # resized的轮廓
        cv2.imshow('ROI', resized_roi)  # ROI区域
        cv2.imshow('result', result_image)  # 标签匹配区域
        key = cv2.waitKey(10)  # 停顿10ms后进入下一次循环
        if key == 32:  # 如果检测到按下空格键
            break

    # 银矿石检测：
    if user_input == "b":
        resized_img = cv2.resize(image, (500, 500))
        hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
        silver_mask = cv2.inRange(hsv_img, silver_lower, silver_upper)
        mask = cv2.inRange(hsv_img, silver_lower, silver_upper)  # 创建掩模
        silver_img = cv2.bitwise_and(resized_img, resized_img, mask=mask)
        silver_img[np.where(mask == 0)] = [0, 0, 0]  # 将非黄色区域设置为黑色
        gray = cv2.cvtColor(silver_img, cv2.COLOR_BGR2GRAY)
        s, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            black_img = np.zeros((500, 500, 3), np.uint8)
            resized_img = black_img
            r1 = black_img
            resized_roi = black_img
            result_image = black_img
        else:
            max_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(max_contour)
            w1, h1 = rect[1]
            area = w1 * h1
            if area < 1000:
                print("未检测到矿石！")
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            x, y, w, h = cv2.boundingRect(box)
            rotated_roi = resized_img[y:y + h, x:x + w]
            if rotated_roi.size != 0:
                scale_factor = 2
                scaled_roi = cv2.resize(rotated_roi, None, fx=scale_factor, fy=scale_factor,
                                        interpolation=cv2.INTER_LINEAR)
                resized_roi = cv2.resize(scaled_roi, (500, 500))
            else:
                width, height = 500, 500
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                blank[1, 1] = (255, 255, 255)
                resized_roi = blank
            ROI = resized_roi
            gray2 = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            _, binary2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)
            r1 = cv2.Canny(binary2, 100, 255)
            contours2, _ = cv2.findContours(r1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            highlighted_image = np.zeros_like(resized_roi)
            for contour in contours2:
                area = cv2.contourArea(contour)
                if 1000 <= area <= 8000:
                    x, y, w, h = cv2.boundingRect(contour)
                    if x <= resized_roi.shape[1] // 2 and y <= resized_roi.shape[0] // 2:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            print('left_arrow_sign get')
                            left_arrow_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            print('left_square_sign get')
                            left_square_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    if x >= resized_roi.shape[1] // 2 and y <= resized_roi.shape[0] // 2:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            right_arrow_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            right_square_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    if x <= resized_roi.shape[1] // 2 and y >= resized_roi.shape[0] // 2:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            ld_arrow_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            ld_square_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                    if x >= resized_roi.shape[1] // 2 and y >= resized_roi.shape[0] // 2:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        x2, y2, w2, h2 = cv2.boundingRect(contour)
                        aspect_ratio = w2 / float(h2)
                        if len(approx) == 6 and 0.8 <= aspect_ratio <= 1.2:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            rd_arrow_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
                        if len(approx) == 4 and 0.75 <= aspect_ratio <= 1.35:
                            cv2.drawContours(resized_roi, [approx], 0, (0, 255, 0), 2)
                            rd_square_sign = 1
                            cv2.drawContours(highlighted_image, [contour], -1, (0, 255, 0), 2)
            result_image = cv2.addWeighted(resized_roi, 0.4, highlighted_image, 0.6, 0)
            cv2.putText(result_image, 'la:' + str(left_arrow_sign), (3, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0),
                        1)
            cv2.putText(result_image, 'ls:' + str(left_square_sign), (3, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'ra:' + str(right_arrow_sign), (3, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'rs:' + str(right_square_sign), (3, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'lds:' + str(ld_square_sign), (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'lda:' + str(ld_arrow_sign), (40, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'rds:' + str(rd_square_sign), (40, 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'rda:' + str(rd_arrow_sign), (40, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (0, 255, 0), 1)
            cv2.putText(result_image, 'posture:' + str(posture), (3, 120), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5,
                        (255, 0, 0), 1)
            if (left_square_sign == 0 and left_arrow_sign == 0) or (right_arrow_sign == 0 and right_square_sign == 0):
                cv2.putText(result_image, 'detecting', (3, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 255), 1)
            if (left_square_sign == 1 or left_arrow_sign == 1) and (right_arrow_sign == 1 or right_square_sign == 1):
                cv2.putText(result_image, 'Gold stone discovery', (3, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 0),
                            1)
                if left_square_sign == 1 and rd_square_sign == 1:
                    left_square_sign = 0
                    rd_square_sign = 0
                    posture = 'rotated'
                if left_arrow_sign == 1 and rd_arrow_sign == 1:
                    left_arrow_sign = 0
                    rd_arrow_sign = 0
                    posture = 'rotated'
                if left_arrow_sign == 1 and rd_square_sign == 1:
                    left_arrow_sign = 0
                    rd_square_sign = 0
                    posture = 'normal'
        cv2.drawContours(resized_img, contours, -1, (0, 0, 255), 2)
        cv2.imshow('silver_img', silver_img)
        cv2.imshow('img', resized_img)
        cv2.imshow('r1', r1)
        cv2.imshow('ROI', resized_roi)
        cv2.imshow('result', result_image)
        key = cv2.waitKey(100)
        if key == 32:
            break

if left_square_sign == 1:
    print('矿石倒置！')
cv2.destroyAllWindows()
capture.release()
