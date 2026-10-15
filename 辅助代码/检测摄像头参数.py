import cv2


def open_camera(index=0, backend=cv2.CAP_DSHOW):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        return None
    return cap


def main():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    cap = None

    for backend in backends:
        print(f"尝试打开摄像头，后端={backend}")
        cap = open_camera(1, backend)
        if cap is not None:
            print(f"摄像头已打开，使用后端={backend}")
            break

    if cap is None or not cap.isOpened():
        print("无法打开摄像头，请检查摄像头是否连接或被占用")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"实际分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    print("按 Esc 键退出")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取画面失败")
            break

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
