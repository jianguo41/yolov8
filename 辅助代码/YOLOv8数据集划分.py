"""YOLOv8 矩形框目标检测数据集划分工具（Python 3.10+，仅标准库）。

运行：python "YOLOv8数据集划分.py"
默认交互输入；若只想修改 CONFIG 后运行，将 interactive 改为 False。
输出：datasets_root/dataset_name/{images,labels}/{train,val} 和 data.yaml。
只复制原始数据，不移动或删除；拒绝向非空输出目录写入，防止新旧划分混杂。
输入为平铺图片目录和同名 YOLO .txt 标签，暂不递归扫描子目录。
仅适用于检测框：class_id x_center y_center width height。
如果图片来自视频，请先按视频/拍摄批次分组；本工具按图片分层，不能防止相邻帧泄漏。
"""

import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path


# ============================================================
# 用户设置：每次处理一个数据集，修改 dataset_name 即可另存一个数据集。
# ============================================================
CONFIG = {
    "datasets_root": r"D:\yolo\yolov8\datasets",  # 所有数据集的总目录
    "dataset_name": "dataset_material_4",                 # 本次输出文件夹名称
    "src_image_dir": r"D:\yolo\material\picture",           # 原脚本中的路径，请按实际修改
    "src_label_dir": r"D:\yolo\material\label",             # None = 自动查找同名标签
    # 与 D:\yolo\txt\material_0.txt 的行顺序一致，类别编号从 0 开始。
    "class_names": ['red','lightblue','darkblue','green','yellow','black','target1','target2','target3'],
    # class_names[0] 对应标签编号 0，依此类推，不能根据当前出现的类别重新编号。
    "stratify_mode": "min_class",                 # min_class / all / single
    "stratify_class": 0,                          # single 模式关注的类别
    "val_ratio": 0.20,
    "random_seed": 91,
    "extensions": {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"},
    "allow_missing_labels": True,                # True = 明确把无标签图片当作背景
    # 空 .txt 表示已确认的背景图，始终允许；丢失标签默认报错，避免误当背景。
    "dry_run": False,                             # True = 只检查和预览，不写文件
    "interactive": True,                          # 首次使用建议保持 True
}


def clean_path(value):
    return str(value).strip().strip('"')


def is_within(child, parent):
    return child == parent or parent in child.parents


def validate_config(cfg):
    root = Path(clean_path(cfg["datasets_root"])).resolve()
    name = str(cfg["dataset_name"]).strip()
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)
    }
    if (not name or name in {".", ".."} or
            any(c in name for c in '<>:"/\\|?*') or
            any(ord(c) < 32 for c in name) or name.endswith((".", " ")) or
            name.split(".")[0].upper() in reserved):
        raise ValueError("dataset_name 必须是合法的单个文件夹名称，不能填写完整路径。")
    dst = (root / name).resolve()
    if dst.parent != root:
        raise ValueError("输出目录必须是 datasets_root 的直接子目录。")
    src = Path(clean_path(cfg["src_image_dir"])).resolve()
    labels = Path(clean_path(cfg["src_label_dir"])).resolve() if cfg["src_label_dir"] else None
    for path in (src, labels):
        if path is None:
            continue
        if not path.is_dir():
            raise NotADirectoryError(f"源目录不存在：{path}")
        if is_within(path, dst) or is_within(dst, path):
            raise ValueError(f"源目录和输出目录不能互相包含：{path} / {dst}")
    if dst.exists() and (not dst.is_dir() or any(dst.iterdir())):
        raise FileExistsError(f"输出目录已存在且非空：{dst}\n请更换 dataset_name，例如 my_dataset_v2。")
    names = cfg["class_names"]
    if not isinstance(names, (list, tuple)) or not names:
        raise ValueError("请填写 class_names，按标注类别编号从 0 开始排列。")
    if any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("每个类别名称必须是非空字符串。")
    if len(set(names)) != len(names):
        raise ValueError("类别名称不能重复。")
    if not 0 < cfg["val_ratio"] < 1:
        raise ValueError("val_ratio 必须大于 0 且小于 1。")
    if cfg["stratify_mode"] not in {"min_class", "all", "single"}:
        raise ValueError("stratify_mode 只能为 min_class、all 或 single。")
    if cfg["stratify_mode"] == "single" and not 0 <= cfg["stratify_class"] < len(names):
        raise ValueError("stratify_class 不在类别编号范围内。")
    return src, labels, dst


def find_label_path(image, label_dir):
    candidates = [label_dir / (image.stem + ".txt")] if label_dir else [
        image.with_suffix(".txt"),
        image.parent.parent / "labels" / (image.stem + ".txt"),
        image.parent / "labels" / (image.stem + ".txt"),
    ]
    found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(found) > 1:
        raise ValueError(f"发现多份同名标签，请明确指定 src_label_dir：{image.name}\n{found}")
    return found[0] if found else None


def parse_label(label, class_count):
    """严格检查检测标签；不静默跳过错误行，不修改或重新编号。"""
    classes = set()
    if label is None:
        return classes
    for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError("检测标签每行必须有 5 列；不支持直接输入分割/姿态标签")
            class_id = int(fields[0])
            if not 0 <= class_id < class_count:
                raise ValueError(f"类别编号应在 0～{class_count - 1} 之间")
            x, y, width, height = map(float, fields[1:])
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y, width, height)):
                raise ValueError("框坐标必须是 0～1 之间的有限数值")
            if width <= 0 or height <= 0:
                raise ValueError("框的宽和高必须大于 0")
            # 给归一化坐标的小数舍入留出极小容差。
            if min(x - width / 2, y - height / 2) < -1e-5 or max(x + width / 2, y + height / 2) > 1 + 1e-5:
                raise ValueError("检测框超出图片边界，请检查标注")
        except ValueError as exc:
            raise ValueError(f"{label} 第 {line_no} 行：{exc}") from exc
        classes.add(class_id)
    return classes


def stratum_key(classes, mode, target):
    if not classes:
        return ("background",)
    if mode == "all":
        return ("all", *sorted(classes))
    if mode == "single":
        return ("single", int(target in classes))
    return ("min_class", min(classes))


def collect_samples(src, label_dir, cfg):
    extensions = {e.lower() for e in cfg["extensions"]}
    images = sorted((p for p in src.iterdir() if p.is_file() and p.suffix.lower() in extensions),
                    key=lambda p: (p.name.casefold(), p.name))
    if not images:
        raise ValueError(f"目录中没有支持的图片（本工具不递归扫描）：{src}")
    samples, stems, missing = [], set(), []
    for image in images:
        if image.stem.casefold() in stems:
            raise ValueError(f"有相同主文件名的图片，无法唯一配对标签：{image.name}")
        stems.add(image.stem.casefold())
        label = find_label_path(image, label_dir)
        if label is None:
            missing.append(image.name)
        classes = parse_label(label, len(cfg["class_names"]))
        samples.append((image, label, classes))
    if missing and not cfg["allow_missing_labels"]:
        raise ValueError(f"{len(missing)} 张图片缺少标签，例如：{', '.join(missing[:8])}\n"
                         "请补齐标注；仅当确认这些图片没有目标时，设置 allow_missing_labels=True。")
    if missing:
        print(f"注意：{len(missing)} 张缺标签图片将作为背景，并生成空标签。")
    if not any(s[2] for s in samples):
        raise ValueError("所有图片均无目标，不能用于本次目标检测训练。")
    return samples


def stratified_split(samples, cfg):
    groups = defaultdict(list)
    for index, (_, _, classes) in enumerate(samples):
        groups[stratum_key(classes, cfg["stratify_mode"], cfg["stratify_class"])].append(index)
    rng = random.Random(cfg["random_seed"])
    train, val = [], []
    for key in sorted(groups):
        indices = groups[key][:]
        rng.shuffle(indices)
        # 保留原脚本的原则：每层至少留 1 张训练；单样本层全部留作训练。
        count = min(len(indices) - 1, max(1, round(len(indices) * cfg["val_ratio"])))
        val.extend(indices[:count])
        train.extend(indices[count:])
    if not train or not val:
        raise ValueError("无法得到非空训练集和验证集。请增加样本或调整分层策略；单样本层不会分入验证集。")
    return sorted(train), sorted(val)


def split_and_copy(cfg):
    src, labels, dst = validate_config(cfg)
    samples = collect_samples(src, labels, cfg)
    train, val = stratified_split(samples, cfg)
    print(f"\n输出：{dst}")
    print(f"总数：{len(samples)}；训练：{len(train)}；验证：{len(val)}")
    print(f"目标验证比例：{cfg['val_ratio']:.1%}；实际：{len(val) / len(samples):.1%}")
    print("分层会使实际比例有所偏差，且不能保证每个类别都出现在两部分。")
    for class_id, name in enumerate(cfg["class_names"]):
        counts = [sum(class_id in samples[i][2] for i in indices) for indices in (train, val)]
        print(f"  {class_id} {name}：训练 {counts[0]} 张，验证 {counts[1]} 张")
        if 0 in counts:
            print("    注意：该类别在训练集或验证集中缺失，请补充数据或检查划分。")
    if not any(samples[i][2] for i in train) or not any(samples[i][2] for i in val):
        raise ValueError("训练集或验证集完全没有目标。请补充目标图片或调整分层策略。")
    yaml_text = (
        "# YOLOv8 目标检测数据集；类别顺序必须与原始标签一致。\n"
        f"path: {json.dumps(dst.as_posix(), ensure_ascii=False)}\n"
        "train: images/train\nval: images/val\n"
        f"nc: {len(cfg['class_names'])}\nnames:\n" +
        "".join(f"  {i}: {json.dumps(name, ensure_ascii=False)}\n"
                for i, name in enumerate(cfg["class_names"]))
    )
    print("\n将生成 data.yaml：\n" + yaml_text)
    if cfg["dry_run"]:
        print("预览结束，没有创建目录或复制文件。")
        return dst
    for split, indices in (("train", train), ("val", val)):
        image_out = dst / "images" / split
        label_out = dst / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for i in indices:
            image, label, _ = samples[i]
            # 独占创建文件，不覆盖已有文件。
            with image.open("rb") as source, (image_out / image.name).open("xb") as target:
                shutil.copyfileobj(source, target)
            with (label_out / (image.stem + ".txt")).open("x", encoding="utf-8") as target:
                if label is not None:
                    target.write(label.read_text(encoding="utf-8-sig"))
    with (dst / "data.yaml").open("x", encoding="utf-8") as target:
        target.write(yaml_text)
    manifest = {
        "source_images": str(src), "source_labels": str(labels) if labels else None,
        "class_names": cfg["class_names"], "seed": cfg["random_seed"],
        "stratify_mode": cfg["stratify_mode"], "stratify_class": cfg["stratify_class"],
        "requested_val_ratio": cfg["val_ratio"],
        "train": [samples[i][0].name for i in train],
        "val": [samples[i][0].name for i in val],
    }
    with (dst / "split_manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    print("\n划分完成，原始图片和标签未改动。训练检查命令：")
    print(f'yolo detect train model=yolov8n.pt data="{(dst / "data.yaml").as_posix()}" '
          'epochs=1 imgsz=640 batch=4 device=0 workers=0')
    return dst


def ask(prompt, default):
    return input(f"{prompt} [{default}]：").strip() or str(default)


def main():
    cfg = CONFIG.copy()
    try:
        if cfg["interactive"]:
            print("YOLOv8 数据集划分；回车保留默认值。")
            print(f"数据集总目录：{cfg['datasets_root']}")
            cfg["dataset_name"] = ask("本次数据集名称", cfg["dataset_name"])
            cfg["src_image_dir"] = clean_path(ask("源图片目录", cfg["src_image_dir"]))
            label = ask("源标签目录（输入 auto 自动查找）", cfg["src_label_dir"] or "auto")
            cfg["src_label_dir"] = None if label.lower() == "auto" else clean_path(label)
            names = ask("类别名称（按编号顺序，用英文逗号分隔）", ",".join(cfg["class_names"]))
            cfg["class_names"] = [n.strip() for n in names.split(",") if n.strip()]
            cfg["val_ratio"] = float(ask("验证集比例", cfg["val_ratio"]))
            cfg["random_seed"] = int(ask("随机种子", cfg["random_seed"]))
            cfg["stratify_mode"] = ask("分层策略 min_class/all/single", cfg["stratify_mode"])
            if cfg["stratify_mode"] == "single":
                cfg["stratify_class"] = int(ask("关注的类别编号", cfg["stratify_class"]))
            cfg["allow_missing_labels"] = ask("把缺失标签视为背景？y/n", "y" if cfg["allow_missing_labels"] else "n").lower() == "y"
            cfg["dry_run"] = ask("仅预览，不复制？y/n", "y" if cfg["dry_run"] else "n").lower() == "y"
        split_and_copy(cfg)
    except (OSError, ValueError) as exc:
        print(f"\n未完成：{exc}")
        print("如果复制期间中断，已复制的文件会保留；请检查后换一个输出名称重试。")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
