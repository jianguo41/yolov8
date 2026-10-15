import sys

import cv2

# 创建摄像头对象，0=第一个USB摄像头，多摄像头可试1、2
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# 设置分辨率720P（1280×720）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# 判断是否成功打开
if not cap.isOpened():
    print("无法打开摄像头！检查USB接线/设备索引")
    sys.exit()

print("摄像头已开启，按 q 退出窗口")

while True:
    # 读取一帧画面
    ret, frame = cap.read()
    if not ret:
        print("读取画面失败")
        break

    # 显示画面窗口
    cv2.imshow("720P Camera", frame)

    # 按q键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
