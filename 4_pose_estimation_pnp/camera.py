import cv2
from datetime import datetime

# 打开默认摄像头（0 表示默认摄像头，1 表示外接摄像头）
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" 无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print(" 无法读取画面")
        break

    # 显示摄像头画面
    cv2.imshow("Press SPACE to Capture / ESC to Exit", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # 按下 ESC 退出
        break
    elif key == 32:  # 按下空格拍照
        # 保存带时间戳的图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f" 已保存：{filename}")

cap.release()
cv2.destroyAllWindows()
