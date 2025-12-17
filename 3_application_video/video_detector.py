import cv2
import numpy as np
from datetime import datetime


def detect_blue_light_arrow_v3(frame):
    """
    利用 2 v3 版本：使用最小旋转矩形框选蓝色灯条目标
    """

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 30, 180], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_frame = frame.copy()
    results = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        cv2.drawContours(result_frame, [cnt], -1, (0, 255, 0), 2)

        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        if w < 5 or h < 5:
            continue

        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(result_frame, [box], 0, (0, 0, 255), 2)

        results.append(((int(x), int(y)), (int(w), int(h)), angle))

    return mask, result_frame, results


def process_video(video_path):
    """
    打开视频文件或摄像头，实时检测蓝色灯条
    支持：
      - 空格键暂停/继续
      - S 键保存当前检测帧（带时间戳，不退出）
      - ESC 键退出
      - 输出 MP4 视频
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("无法打开视频文件或摄像头，请检查路径。")
        return

    # 获取视频信息
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25

    print("视频已打开：")
    print("   [空格] 暂停/继续播放")
    print("   [S] 保存当前检测帧")
    print("   [ESC] 提前退出程序")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"output_arrow_detect_v3_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_name, fourcc, fps, (w, h))
    print(f"正在录制输出视频（如果提前退出将只有运行部分）：{out_name}")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频播放结束或无法读取帧。")
                break

            # 调用检测函数
            mask, result_frame, results = detect_blue_light_arrow_v3(frame)

            # 写入输出视频
            out.write(result_frame)

        cv2.imshow("Blue Light Detection (v3)", result_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC 键退出
            print("已退出视频检测。")
            break

        elif key == ord('s'):  # S 键保存帧
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"arrow_detect_v3_{ts}.png"
            cv2.imwrite(save_name, result_frame)
            print(f"已保存检测帧：{save_name}")

        elif key == ord(' '):  # 空格键暂停/继续
            paused = not paused
            if paused:
                print("⏸ 暂停中... 再按空格继续。")
            else:
                print("▶ 继续播放...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已保存至：{out_name}")



if __name__ == "__main__":
    video_path = "video.mp4"

    if video_path == "0":
        video_path = 0

    process_video(video_path)
