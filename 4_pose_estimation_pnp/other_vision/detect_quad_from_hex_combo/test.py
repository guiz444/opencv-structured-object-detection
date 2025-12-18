import cv2
import numpy as np
import itertools
"""

功能：
    - 检测黄色区域内的黑色目标（方形和六边形）并分类
    - 根据 1 个方形 + 3 个六边形组合寻找四点标记
    - 提供几何评分，选择最接近矩形的组合
    - 支持连续帧平滑与跳动过滤
    - 可视化质心、类型、编号及连线

依赖：
    - OpenCV (cv2)
    - NumPy (np)
    - Python 3.6 以上

使用：
    python detect_black_shapes_on_yellow.py
    - q：退出
    - 窗口显示检测结果与标记
"""

def get_centroid(contour, offset=(0, 0)):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"]) + offset[0]
    cy = int(M["m01"] / M["m00"]) + offset[1]
    return (cx, cy)

def sort_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sorted_pts = pts[np.argsort(angles)]
    return sorted_pts

def quad_geometry_score(pts):
    """评分：越小越接近矩形"""
    pts = np.array(pts, dtype=np.float32)
    mid_ac = (pts[0] + pts[2]) / 2
    mid_bd = (pts[1] + pts[3]) / 2
    center_diff = np.linalg.norm(mid_ac - mid_bd)
    edges = np.diff(np.vstack([pts, pts[0]]), axis=0)
    norms = np.linalg.norm(edges, axis=1) + 1e-6
    orth_errs = []
    for i in range(4):
        a = edges[i]
        b = edges[(i+1)%4]
        cos = abs(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))
        orth_errs.append(cos)
    orth_error = np.mean(orth_errs)
    diag1 = np.linalg.norm(pts[0] - pts[2])
    diag2 = np.linalg.norm(pts[1] - pts[3])
    diag_diff = abs(diag1 - diag2)
    side_lengths = norms
    side_ratio = (np.max(side_lengths) / (np.min(side_lengths) + 1e-6))
    score = center_diff + 0.7 * orth_error * 100 + 0.2 * diag_diff + 5.0 * (side_ratio - 1.0)
    return score

def detect_black_shapes_on_yellow_with_types(frame, prev_pts=None, params=None):
    if params is None:
        params = {}
    min_area = params.get('min_area', 100)
    rectangle_score_thresh = params.get('score_thresh', 50)
    max_frame_jump = params.get('max_frame_jump', 80)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([15, 180, 160])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

    contours_y, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = frame.copy()
    if not contours_y:
        return vis, prev_pts, prev_pts

    yellow_contour = max(contours_y, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(yellow_contour)
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])
    black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
    contours_b, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    for cnt in contours_b:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        sides = len(approx)
        c = get_centroid(cnt, (x, y))
        if c is None:
            continue
        if sides == 4:
            typ = 'square'
        elif 5 <= sides <= 7:
            typ = 'hex'
        else:
            continue
        shapes.append({'pt': (int(c[0]), int(c[1])), 'type': typ, 'area': area})

    # 可视化检测点
    for s in shapes:
        cv2.circle(vis, s['pt'], 4, (200,200,200), -1)
        cv2.putText(vis, s['type'][0].upper(), (s['pt'][0]+3, s['pt'][1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    num_squares = sum(1 for s in shapes if s['type']=='square')
    num_hexes = sum(1 for s in shapes if s['type']=='hex')
    best_quad, best_score = None, 1e9

    if num_squares >= 1 and num_hexes >= 3:
        squares_idx = [i for i,s in enumerate(shapes) if s['type']=='square']
        hex_idx = [i for i,s in enumerate(shapes) if s['type']=='hex']
        for si in squares_idx:
            sq_pt = shapes[si]['pt']
            hex_pts = [shapes[i]['pt'] for i in hex_idx]
            for combo in itertools.combinations(hex_idx, 3):
                pts = [sq_pt] + [shapes[i]['pt'] for i in combo]
                pts_arr = sort_points_clockwise(pts)
                score = quad_geometry_score(pts_arr)
                if score < best_score:
                    best_score = score
                    best_quad = np.array(pts_arr, dtype=np.float32)

    selected_quad = None
    if best_quad is not None and best_score < rectangle_score_thresh:
        selected_quad = best_quad
    else:
        selected_quad = None

    # 稳定逻辑
    if selected_quad is not None:
        if prev_pts is not None:
            prev_center = np.mean(prev_pts, axis=0)
            curr_center = np.mean(selected_quad, axis=0)
            move_dist = np.linalg.norm(curr_center - prev_center)
            if move_dist > max_frame_jump:
                selected_quad = prev_pts.copy()
        prev_pts = np.array(selected_quad, dtype=np.float32)
    else:
        if prev_pts is not None:
            selected_quad = prev_pts.copy()

    # 可视化
    if selected_quad is not None:
        quad_i = np.int32(selected_quad)
        for i,p in enumerate(quad_i):
            cv2.circle(vis, tuple(p), 6, (0,255,0), -1)
            cv2.putText(vis, str(i+1), (p[0]+5, p[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.line(vis, tuple(quad_i[i]), tuple(quad_i[(i+1)%4]), (255,0,0), 2)

    return vis, (selected_quad.astype(np.int32) if selected_quad is not None else None), prev_pts


# ========== 主循环 ==========
if __name__ == "__main__":
    cap = cv2.VideoCapture("stone.mp4")
    prev_pts = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        vis, quad, prev_pts = detect_black_shapes_on_yellow_with_types(
            frame, prev_pts,
            params={'min_area':100, 'score_thresh':50, 'max_frame_jump':80}
        )

        if quad is not None:
            # 输出质心坐标
            print("✅ 当前四个多边形质心（按顺时针）:")
            for i, (x, y) in enumerate(quad):
                print(f"点{i+1}: ({x}, {y})")
        else:
            print("未检测到满足 1 square + 3 hex 的矩形")

        cv2.imshow("Gold Stone Marker Detection", vis)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
