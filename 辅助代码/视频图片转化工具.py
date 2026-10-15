import cv2
import os

# ========== 请修改这里 ==========
video_path = r"D:\yolo\videos\obstacle.mp4"  # 你的视频路径
save_dir = r"D:\yolo\pictures"  # 保存图片的文件夹
file_prefix = "obs_"  # 输出图片的前缀，由用户决定
N = 8 # 每 N 帧保存一张
# ===============================

os.makedirs(save_dir, exist_ok=True)
print(f"输出图片文件夹：{save_dir}")

cap = cv2.VideoCapture(video_path)

frame_idx = 0
img_idx = 1  # 图片从 1 开始编号

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % N == 0:
        save_path = os.path.join(save_dir, f"{file_prefix}{img_idx}.jpg")
        cv2.imwrite(save_path, frame)
        img_idx += 1

    frame_idx += 1

cap.release()
print(f"保存完成！共 {img_idx - 1} 张图片，文件名：{file_prefix}1.jpg、{file_prefix}2.jpg、{file_prefix}3.jpg……")