from pathlib import Path

from ultralytics import YOLO


# ============================================================
# 断点续训配置区
#
# 将 CHECKPOINT 改为被中断实验的 weights/last.pt 完整路径。
# 必须使用 last.pt：它保存了 epoch、优化器和学习率等训练状态。
# best.pt 只保存表现最好的模型权重，不能完整恢复中断前的训练过程。
# ============================================================

CHECKPOINT = r"D:\yolo\yolov8\runs\detect\material_1\weights\last.pt"


# ============================================================
# 核心逻辑：一般不需要修改
# ============================================================

def main():
    """从 last.pt 恢复同一个实验的训练。"""

    checkpoint = Path(CHECKPOINT).expanduser().resolve()

    # 路径和文件名检查，避免把不存在的路径或 best.pt 当成续训断点。
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到断点权重：{checkpoint}")
    if checkpoint.name.lower() != "last.pt":
        raise ValueError(
            "续训请填写 weights/last.pt；best.pt 用于预测或重新微调，"
            "不能恢复中断时的优化器和训练轮数。"
        )

    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError(f"当前模型任务为 {model.task}，此脚本只用于目标检测续训。")

    print(f"从断点继续训练：{checkpoint}")
    print("将恢复原训练的模型、优化器、学习率和 epoch；无需重新填写 data、batch、epochs 等参数。")

    # resume=True 会读取 last.pt 内保存的原实验参数和训练状态。
    model.train(resume=True)

    save_dir = Path(model.trainer.save_dir).resolve()
    print("\n续训结束！")
    print(f"结果目录：{save_dir}")
    for filename in ("best.pt", "last.pt"):
        path = save_dir / "weights" / filename
        if path.is_file():
            print(f"{filename}：{path}")


# Windows 下保留入口保护，避免训练进程重复启动。
if __name__ == "__main__":
    main()
