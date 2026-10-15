from pathlib import Path

from ultralytics import YOLO

# ============================================================
# 用户配置区
# 每次实验主要修改：data、model、project、name。
#
# 路径前的 r 表示原始字符串，适合填写 Windows 路径。
# 下方以当前数据集为示例，以后直接替换路径即可。
# ============================================================

CONFIG = {
    # ---------------- 文件路径 ----------------
    # 数据集 YAML 文件的完整路径。
    # 类别数量、类别名称和图片路径从这个文件中读取。
    "data": r"D:\yolo\yolov8\datasets\dataset_obstacle\data.yaml",
    # 模型名称或权重路径。
    # 首次训练一般使用预训练权重，例如 yolov8n.pt、yolov8s.pt。
    # 也可填写自己训练出的 best.pt，在其基础上开始新的微调。
    # n/s/m/l/x 模型逐渐增大，计算量和显存需求通常也增加。
    "model": r"D:\yolo\yolov8\yolov8n.pt",
    # 所有实验结果的保存目录。
    "project": r"D:\yolo\yolov8\runs\detect",
    # 本次实验的文件夹名称，可以自行修改。
    "name": "obstacle_0",
    # ---------------- 基本训练参数 ----------------
    # 最大训练轮数。一轮表示遍历一次训练集。
    # 首次验证数据和流程时，可设为 1。
    "epochs": 200,
    # 输入尺寸。程序会处理缩放与填充，不需要手动拉伸原图。
    # 增大尺寸可能改善小目标识别，但会增加显存和计算开销。
    "imgsz": 640,
    # 每批图片数量。显存不足时减小，例如 4 改成 2。
    # 这是初始设置，不代表所有任务的最佳值。
    "batch": 4,
    # 0 = 第一张 NVIDIA GPU；"cpu" = 使用 CPU。
    "device": 0,
    # 数据加载进程数量。
    # Windows 首次运行建议设为 0，跑通后可尝试 2 或 4。
    "workers": 0,
    # 验证表现连续多少轮没有改善时提前停止。
    # 不是根据训练损失是否下降判断。
    "patience": 30,
    # 随机种子，便于比较实验。
    # 相同种子不保证跨硬件、跨软件版本完全复现。
    "seed": 42,
    # ---------------- 性能与输出 ----------------
    # 自动混合精度，通常可以降低显存占用并加快 GPU 训练。
    # 遇到相关数值异常时，可以设为 False 排查。
    "amp": True,
    # 尽量使用确定性计算，可能影响速度。
    "deterministic": True,
    # 图片缓存：False 不启用；"ram" 缓存到内存；
    # "disk" 缓存到磁盘。内存有限时先保持 False。
    "cache": False,
    # 训练期间执行验证集评估。
    "val": True,
    # 保存模型权重。
    "save": True,
    # 保存曲线、训练图片和验证结果等可视化输出。
    "plots": True,
    # False：同名实验目录存在时自动增加编号。
    # True：允许使用已有目录，可能覆盖输出文件。
    "exist_ok": False,
}


# ============================================================
# 可选参数区
#
# 默认全部注释，由当前 Ultralytics 版本使用其默认设置。
# 有需要时取消相应行开头的 #。
# 此处同名参数会覆盖 CONFIG 中的值。
#
# 建议一次只调整少量参数，便于判断效果变化的原因。
# ============================================================

EXTRA_ARGS = {
    # ---------------- 优化器和学习率 ----------------
    # "auto" 自动选择优化器；也可指定 "AdamW" 或 "SGD"。
    # 自动选择策略可能随 Ultralytics 版本变化。
    # "optimizer": "auto",
    # 手动研究学习率时，可以指定优化器后再设置 lr0。
    # 使用 auto 时，程序可能覆盖手动设置的学习率、动量。
    # "optimizer": "AdamW",
    # "lr0": 0.001,          # 初始学习率
    # "lrf": 0.01,          # 最终学习率相对于初始值的比例
    # 权重衰减：正则化参数，可帮助控制过拟合。
    # "weight_decay": 0.0005,
    # SGD 中的动量；Adam 类优化器中通常对应第一阶矩系数。
    # "momentum": 0.937,
    # 初期预热轮数，用于逐渐调整学习率等。
    # "warmup_epochs": 3.0,
    # 使用余弦学习率衰减。
    # "cos_lr": True,
    # ---------------- 颜色和光照增强 ----------------
    # 以下是增强设置示例，不保证适合所有任务。
    # 如果颜色决定类别，应谨慎调整色调和饱和度。
    # "hsv_h": 0.015,       # 色调扰动幅度；0.0 表示关闭色调扰动
    # "hsv_s": 0.7,         # 饱和度扰动幅度
    # "hsv_v": 0.4,         # 明暗扰动幅度
    # ---------------- 几何增强 ----------------
    # "degrees": 0.0,       # 随机旋转角度幅度；10.0 约表示 ±10°
    # "translate": 0.1,     # 平移幅度，相对于图片尺寸
    # "scale": 0.5,         # 随机缩放幅度，通常约对应 0.5～1.5 倍
    # "shear": 0.0,         # 随机剪切角度幅度
    # "perspective": 0.0,   # 透视变换幅度，一般使用很小的值
    # "fliplr": 0.5,        # 水平翻转概率
    # "flipud": 0.0,        # 垂直翻转概率
    # 翻转、旋转应符合实际场景。
    # 例如文字或带方向含义的目标，不适合随意翻转。
    # ---------------- 图片混合增强 ----------------
    # Mosaic：将多张图片拼接，增加目标尺度和背景变化。
    # "mosaic": 1.0,        # Mosaic 启用概率
    # "close_mosaic": 10,   # 最后多少轮关闭 Mosaic
    # MixUp：将图片混合叠加。
    # "mixup": 0.0,         # MixUp 启用概率
    # ---------------- 数据与模型范围 ----------------
    # 使用训练集的一部分，方便快速调试。
    # "fraction": 1.0,      # 1.0 为全部，0.5 为约一半
    # 只保留指定类别参与训练。
    # 不会自动修改 YAML 中的类别定义或重新编号。
    # "classes": [0, 1],
    # 将不同类别统一视为一个类别训练。
    # "single_cls": False,
    # 冻结前若干模型层，不更新这些层的参数。
    # 10 表示层索引范围，不是前 10 轮。
    # "freeze": 10,
    # ---------------- 检查点与日志 ----------------
    # 每隔多少轮额外保存一次权重，增加磁盘占用。
    # -1 表示不额外按固定轮数保存。
    # "save_period": 10,
    # 输出更多日志信息。
    # "verbose": True,
}


# ============================================================
# 断点续训配置
#
# 新训练：保持 None。
# 意外中断后继续原训练：填写该实验的 weights/last.pt。
#
# 注意：
# 断点续训会恢复原实验保存的配置、轮数和优化器状态，
# 不按上方 CONFIG 开始一个新实验。
# 已正常结束的训练若要继续微调：
# 保持 RESUME_CHECKPOINT=None，把 CONFIG["model"] 改为 best.pt。
# ============================================================

RESUME_CHECKPOINT = None

# 例：
# RESUME_CHECKPOINT = (
#     r"D:\yolo\yolov8\runs\detect\experiment_001\weights\last.pt"
# )


# ============================================================
# 核心逻辑：日常使用一般不需要修改
# ============================================================


def train_new():
    """开始新的训练或微调实验。."""
    # 合并参数，可选参数覆盖基础配置中的同名项。
    args = {**CONFIG, **EXTRA_ARGS}

    data_path = Path(args["data"]).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"找不到数据集配置文件：{data_path}")

    args["data"] = str(data_path)

    # model 用于创建模型，不重复传入 train()。
    model_path = args.pop("model")
    model = YOLO(model_path)

    # 此模板用于矩形框目标检测。
    if model.task != "detect":
        raise ValueError(f"当前模型任务为 {model.task}，本模板请使用 YOLOv8 目标检测模型，例如 yolov8n.pt。")

    print(f"数据配置：{args['data']}")
    print(f"初始模型：{model_path}")
    print(f"实验名称：{args['name']}")

    model.train(**args)
    return model


def resume_training():
    """从中断实验的检查点恢复训练。."""
    checkpoint = Path(RESUME_CHECKPOINT).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到断点权重：{checkpoint}")

    print(f"恢复原训练：{checkpoint}")
    print("本次将使用检查点保存的训练配置。")

    model = YOLO(str(checkpoint))
    model.train(resume=True)
    return model


def main():
    if RESUME_CHECKPOINT:
        model = resume_training()
    else:
        model = train_new()

    # 读取实际输出目录，兼容同名实验自动编号的情况。
    save_dir = Path(model.trainer.save_dir).resolve()

    print("\n训练结束！")
    print(f"结果目录：{save_dir}")

    # 如果用户关闭保存或使用了特殊设置，不宣称不存在的文件已生成。
    for filename in ("best.pt", "last.pt"):
        weight_path = save_dir / "weights" / filename
        if weight_path.is_file():
            print(f"{filename}：{weight_path}")

    # 实际使用的训练参数通常会记录在结果目录的 args.yaml 中。


# Windows 下保留入口保护，避免多进程重复执行训练。
if __name__ == "__main__":
    main()
