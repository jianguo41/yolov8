import cv2

cap = cv2.VideoCapture("/dev/camera1", cv2.CAP_V4L2)

try:
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头 /dev/camera1")

    # 先设置 MJPG：你的摄像头在此格式下支持 1280×720、30 FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    # 请求分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 打印摄像头实际报告的参数
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    pixel_format = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    print("采集格式：", pixel_format)
    print("分辨率：", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "×", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("摄像头报告帧率：", cap.get(cv2.CAP_PROP_FPS))
    print("在预览窗口按 q 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("摄像头读取失败")
            break

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\n已停止")

finally:
    cap.release()
    cv2.destroyAllWindows()
