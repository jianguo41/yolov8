"""通用 YOLOv8 目标检测预测模板。.

运行：conda activate yolov8 后执行 python predict_yolov8.py
支持：单张图片、图片目录、本地视频、摄像头、网络视频流。
日常只修改 CONFIG / EXTRA_ARGS，不需要修改 main()。
模型中的类别来自权重文件，预测不需要提供训练用的 data.yaml。
"""

from pathlib import Path

from ultralytics import YOLO

# ============================================================
# 一、基本配置：换任务时主要修改 model、source、project、name
# Windows 路径前加 r，防止反斜杠被解析为转义字符。
# ============================================================

CONFIG = {
    # 模型权重路径。
    # 用自己的 best.pt：检测自己训练的类别。
    # 用官方 yolov8n.pt：检测 COCO 预训练的类别。
    "model": r"D:\yolo\yolov8\runs\detect\material_0\weights\best.pt",
    # 检测来源，每次选择一种。
    # 本地视频：r"D:\yolo\videos_red\红色物料2.mp4"
    # 单张图片：r"D:\yolo\test.jpg"
    # 图片目录：r"D:\yolo\test_images"
    # 默认摄像头：0（整数，不加引号）；其他摄像头可尝试 1。
    # 网络视频流："rtsp://摄像头地址/具体流路径"
    "source": 0,
    # 结果保存的总目录。
    "project": r"D:\yolo\yolov8\runs\predict",
    # 本次预测的文件夹名称，可自行修改，不填写完整路径。
    "name": "red_video_experiment2",
    # 是否保存画有检测框的图片或视频。
    # 保存视频不是逐帧导出图片，默认会生成视频文件。
    "save": False,
    # 是否弹出检测窗口。
    # 文件批量处理可以设为 False；摄像头实时查看通常设为 True。
    "show": True,
    # 同名结果目录存在时：False 自动增加编号；True 允许复用目录。
    # 为避免覆盖旧结果，建议保持 False。
    "exist_ok": False,
    # 推理设备：0 为第一张 NVIDIA GPU；"cpu" 为 CPU。
    # device=0 是显卡编号，source=0 是摄像头编号，二者独立。
    "device": 0,
    # 模型推理输入尺寸，不是强制将输出视频改成 640×640。
    # 增大尺寸可能帮助小目标检测，但更慢、显存开销更大。
    # 通常先与训练尺寸一致。
    "imgsz": 640,
    # 置信度过滤阈值，范围 0～1。
    # 提高：通常减少低置信度误检，也可能增加漏检。
    # 降低：保留更多候选框，也可能增加误检。
    "conf": 0.7,
    # NMS 的 IoU 阈值，用于去除重复检测框。
    # 降低：更容易压制重叠框；提高：允许更多重叠框保留。
    # 这是预测框之间的去重参数，不是验证时 mAP 的 IoU 阈值。
    "iou": 0.7,
    # 每张图片（每帧）最多保留的检测框数量。
    "max_det": 300,
    # 半精度推理。支持的 GPU 后端上可能更快、更省显存。
    # 先使用 False 验证效果；GPU 上可另做 True 的对照实验。
    # 使用 CPU 时保持 False。
    "half": False,
    # 视频抽帧步长：1 表示处理每帧，2 表示每隔一帧处理一次。
    # 抽帧可能漏掉短暂目标，并影响保存视频的时长/播放节奏。
    # 需要完整保存检测视频时保持 1。
    "vid_stride": 1,
    # 实时流缓冲：False 倾向处理新帧、降低延迟；True 排队处理。
    # 若推理跟不上摄像头，True 可能积累延迟。
    # 这不是本地视频是否抽帧的开关。
    "stream_buffer": False,
    # 是否在图片/视频上绘制框、类别名称和置信度。
    "show_boxes": True,
    "show_labels": True,
    "show_conf": True,
    # 检测框线宽，None 为自动；也可手动设置 1、2、3 等。
    "line_width": None,
    # 输出每张图片/每帧的检测信息和耗时；长视频可设为 False。
    "verbose": True,
}


# ============================================================
# 二、可选参数：需要时取消注释；同名项会覆盖 CONFIG 中的设置。
# 没有启用的参数由当前 Ultralytics 版本使用默认值。
# ============================================================

EXTRA_ARGS = {
    # 仅显示指定类别。编号以 model.names 为准，不是随意填写。
    # None 表示所有类别，单类别模型一般无需设置。
    # "classes": [0],
    # 不区分类别地执行 NMS。
    # True 可能压制不同类别之间重叠的检测框。
    # "agnostic_nms": False,
    # 测试时增强：对输入做额外变换并合并预测，通常更慢。
    # 不保证精度提升，先保持关闭。
    # "augment": False,
    # 将预测框保存为文本文件，位于结果目录的 labels 子目录。
    # 注意这些是模型预测，不是经过人工确认的训练标签。
    # "save_txt": True,
    # 在文本结果中额外保存置信度，需要与 save_txt=True 配合。
    # 启用后每行多一列，不能直接当作五列格式的训练标注。
    # "save_conf": True,
    # 保存检测目标的裁剪图片，目标较多时会产生大量文件。
    # "save_crop": True,
    # 保存视频的可视化帧图片，通常与 save=True 配合。
    # 会占用更多空间；仅保存视频不需要开启此项。
    # "save_frames": True,
}


# ============================================================
# 三、核心逻辑：日常使用无需修改
# ============================================================


def validate_source(source):
    """检查本模板支持的常见来源；摄像头和网络流在打开时验证。."""
    if isinstance(source, int):
        if source < 0:
            raise ValueError("摄像头编号不能小于 0。")
        return
    source_text = str(source)
    if source_text.isdigit() or source_text.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "tcp://")):
        return
    if not Path(source_text).exists():
        raise FileNotFoundError(f"找不到检测来源，请检查 source：{source}")


def main():
    args = {**CONFIG, **EXTRA_ARGS}
    model_path = args.pop("model")
    validate_source(args["source"])

    # YOLO 标准权重名称不存在时，库可能尝试下载。
    # 自己的 best.pt 应填写真实存在的路径。
    model = YOLO(model_path)
    if model.task != "detect":
        raise ValueError(f"当前模型任务为 {model.task}；此模板用于矩形框目标检测。")

    print(f"模型：{model_path}")
    print(f"模型类别：{model.names}")
    print(f"检测来源：{args['source']}")

    # stream=True 返回逐张/逐帧结果的生成器，避免把整个视频的
    # 预测结果一直保存在内存中。这里的 stream 与网络直播不是同一含义。
    # 必须遍历生成器，预测与自动保存才会实际执行。
    results = model.predict(stream=True, **args)
    processed = 0
    interrupted = False
    try:
        for result in results:
            processed += 1

            # 如需接入其他程序，可在这里处理每帧检测数据：
            # boxes = result.boxes.xyxy.cpu().numpy()  # 像素坐标 x1,y1,x2,y2
            # scores = result.boxes.conf.cpu().numpy() # 各框置信度
            # ids = result.boxes.cls.int().cpu().tolist()  # 各框类别编号
            # 不要无限地将 result 追加到列表，否则长视频仍会占用大量内存。

    except KeyboardInterrupt:
        interrupted = True
        print("\n用户停止预测，正在结束视频写入。")
    finally:
        results.close()
        # 正常结束时 Ultralytics 会释放视频写入器；中断时也执行释放，
        # 尽量使已经写入的视频片段可播放。强制结束进程无法保证这一点。
        predictor = model.predictor
        if predictor is not None:
            for writer in predictor.vid_writer.values():
                writer.release()
        if args.get("show"):
            import cv2

            cv2.destroyAllWindows()

    print(f"\n{'预测已停止' if interrupted else '预测流程结束'}，已处理 {processed} 张图片/帧。")
    if processed == 0:
        print("没有处理到图片或帧，请检查摄像头、视频文件或网络流。")
    elif any(args.get(key) for key in ("save", "save_txt", "save_crop")):
        save_dir = Path(model.predictor.save_dir).resolve()
        print(f"实际输出目录：{save_dir}")
        print("请检查目录中的结果文件；Windows 下结果视频可能保存为 .avi。")
        print("Ultralytics 输出的检测视频通常不保留原视频音轨。")
    else:
        print("本次未启用文件保存。")


if __name__ == "__main__":
    main()
