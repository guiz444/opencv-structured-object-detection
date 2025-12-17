import cv2
import numpy as np
from datetime import datetime

# ---------------- 改进版跳动过滤 ----------------
def filter_jumping_squares(squares, prev_squares, jump_threshold=50):
    """
    对每个方块独立判断跳动，超出阈值的不输出
    squares: 当前帧检测到的方块列表 [(x, y, w, h), ...]
    prev_squares: 上一帧方块列表 [(x, y, w, h), ...]
    jump_threshold: 像素阈值
    返回: 过滤后的方块列表
    """
    if prev_squares is None or len(prev_squares) == 0:
        return squares  # 无上一帧，直接返回

    filtered = []
    for i, (x1, y1, w1, h1) in enumerate(squares):
        cx1, cy1 = x1 + w1/2, y1 + h1/2

        # 找上一帧中距离最近的方块
        min_dist = float('inf')
        closest_prev = None
        for (x0, y0, w0, h0) in prev_squares:
            cx0, cy0 = x0 + w0/2, y0 + h0/2
            dist = np.linalg.norm([cx1 - cx0, cy1 - cy0])
            if dist < min_dist:
                min_dist = dist
                closest_prev = (x0, y0, w0, h0)

        if min_dist < jump_threshold:
            filtered.append((x1, y1, w1, h1))
            print(f"[通过] 当前方块{i+1} 距离上一帧最近方块 {min_dist:.1f} < 阈值 {jump_threshold}")
        else:
            print(f"[过滤] 当前方块{i+1} 距离上一帧最近方块 {min_dist:.1f} > 阈值 {jump_threshold}，不输出该方块")

    return filtered


def sort_squares_corners(squares):
    """
    输入: squares = [(x, y, w, h), ...]
    输出: 按左上、左下、右上、右下顺序排序的坐标
    使用质心 + 象限判断
    """
    if len(squares) != 4:
        print(f"[排序跳过] 方块数量不是4个: {len(squares)}")
        return squares  # 不处理非四个方块情况

    centers = [(x + w/2, y + h/2, x, y, w, h) for (x, y, w, h) in squares]
    cX = np.mean([c[0] for c in centers])
    cY = np.mean([c[1] for c in centers])

    lt, lb, rt, rb = [], [], [], []

    for cx, cy, x, y, w, h in centers:
        if cx < cX and cy < cY:
            lt.append((x, y, w, h))
        elif cx < cX and cy >= cY:
            lb.append((x, y, w, h))
        elif cx >= cX and cy < cY:
            rt.append((x, y, w, h))
        elif cx >= cX and cy >= cY:
            rb.append((x, y, w, h))

    lt = sorted(lt, key=lambda r: r[1])[:1]
    lb = sorted(lb, key=lambda r: r[1], reverse=True)[:1]
    rt = sorted(rt, key=lambda r: r[1])[:1]
    rb = sorted(rb, key=lambda r: r[1], reverse=True)[:1]

    return lt + lb + rt + rb


def detect_black_shapes_on_yellow(input_data, prev_squares=None, jump_threshold=10, area_diff_ratio=0.5):
    """
    input_data: str 或 ndarray
    prev_squares: 上一帧四个方块，用于跳动过滤
    jump_threshold: 跳动阈值（像素）
    area_threshold: 面积差异阈值比例，如果相差超过这个比例则不输出坐标
    """
    rectangles = []

    if isinstance(input_data, str):
        img = cv2.imread(input_data)
        if img is None:
            print("无法读取图片")
            return None, []
    elif isinstance(input_data, np.ndarray):
        img = input_data.copy()
    else:
        print("输入必须是图片路径或 ndarray")
        return None, []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 180,160])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未检测到黄色方块")
        return img, []

    yellow_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(yellow_contour)
    yellow_roi = img[y:y+h, x:x+w]

    hsv_roi = cv2.cvtColor(yellow_roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([10,10,10])
    upper_black = np.array([35,155,110])
    black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 3:
            rx, ry, rw, rh = cv2.boundingRect(approx)
            rectangles.append((rx + x, ry + y, rw, rh))

    # 筛选近似正方形
    squares = [r for r in rectangles if 0.8 <= r[2]/r[3] <= 1.2]
    if len(squares) > 4:
        squares = sorted(squares, key=lambda r: r[2]*r[3])
        mid = len(squares)//2
        squares = squares[mid-2:mid+2]

        # 整体面积差异过滤
    if len(squares) == 4:
        areas = [w * h for (_, _, w, h) in squares]
        max_area = max(areas)
        min_area = min(areas)
        if min_area == 0 or (max_area - min_area) / min_area > area_diff_ratio:
            # 面积差异过大，直接丢弃
            print(f"[过滤] 面积差异过大: diff={area_diff_ratio:.2f} > {area_diff_ratio}")
            return img, []

    # 跳动过滤
    if prev_squares is not None and len(prev_squares) == 4 and len(squares) == 4:
        filtered = []
        for i in range(4):
            x1, y1, w1, h1 = squares[i]
            x0, y0, w0, h0 = prev_squares[i]
            cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
            cx0, cy0 = x0 + w0 / 2, y0 + h0 / 2
            dist = np.linalg.norm([cx1 - cx0, cy1 - cy0])
            if dist < jump_threshold:
                filtered.append(squares[i])
            else:
                print(f"[过滤] 方块{i + 1}跳动过大，当前帧不输出该方块")
                # 不 append，直接丢弃
        squares = filtered

    # 排序
    if len(squares) == 4:
        squares = sort_squares_corners(squares)

    # 绘制红框
    result_img = img.copy()
    for rect in squares:
        rx, ry, rw, rh = rect
        cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0,0,255), 2)

    return result_img, squares


# 测试
if __name__ == "__main__":
    video_path = "stone.mp4"  # 视频路径
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img, rects = detect_black_shapes_on_yellow(frame)
        # 显示结果
        cv2.imshow("Detected Black Shapes", result_img)

        # 输出坐标（可选）
        print("红色方块矩形坐标 (x, y, w, h):", rects)

        # 按 q 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()