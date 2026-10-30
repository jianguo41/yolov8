# ==================== 用户配置：运行前修改下面四项 ====================
IMAGE_ROOT = r"D:\yolo\merge\picture"          # 总图片目录，递归读取所有子文件夹
LABEL_ROOT = r"D:\yolo\merge\label"     # 改成实际标签根目录，递归读取所有子文件夹
OUTPUT_PARENT = r"D:\yolo"       # 输出文件夹存放的位置
OUTPUT_NAME = "material"            # 输出文件夹名称，内部只有 picture 和 label
# ===================================================================

from collections import Counter, defaultdict
from pathlib import Path
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_files(root, extensions):
    """读取根目录及各层子目录中的文件，排序使输出编号稳定。"""
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: p.relative_to(root).as_posix().casefold(),
    )


def merge_dataset(image_root, label_root, output_parent, output_name):
    """仅复制能确定配对的图片和标签，自动跳过没有 txt 标签的图片。"""
    image_root = Path(image_root).expanduser().resolve()
    label_root = Path(label_root).expanduser().resolve()
    output_parent = Path(output_parent).expanduser().resolve()
    for root in (image_root, label_root):
        if not root.is_dir():
            raise ValueError(f"输入目录不存在：{root}\n请修改代码最前面的路径配置。")
    if (
        not output_name.strip()
        or output_name in {".", ".."}
        or any(c in output_name for c in '<>:"/\\|?*')
        or output_name.endswith((".", " "))
    ):
        raise ValueError("OUTPUT_NAME 必须是合法的单个文件夹名称。")
    output_root = (output_parent / output_name).resolve()
    print(f"计划输出位置：{output_root}")
    if any(output_root == root or root in output_root.parents for root in (image_root, label_root)):
        raise ValueError("输出目录不能位于输入目录内部，请修改 OUTPUT_PARENT。")
    if output_root.exists():
        raise ValueError(f"输出目录已存在，请更换 OUTPUT_NAME，以免覆盖原结果：{output_root}")

    images = collect_files(image_root, IMAGE_EXTENSIONS)
    labels = collect_files(label_root, {".txt"})
    if not images:
        raise ValueError(f"图片根目录及其子目录中没有找到支持的图片：{image_root}")

    # 优先匹配相同相对路径：图片根目录/1/a.jpg -> 标签根目录/1/a.txt。
    # 若目录结构不同，只有图片和标签两边的文件主名都唯一，才按主名匹配。
    # 这可避免把另一个批次的同名标签误配给没有标注的图片。
    by_path = defaultdict(list)
    by_stem = defaultdict(list)
    image_stem_counts = Counter(image.stem.casefold() for image in images)
    for label in labels:
        key = label.relative_to(label_root).with_suffix("").as_posix().casefold()
        by_path[key].append(label)
        by_stem[label.stem.casefold()].append(label)

    pairs, skipped, errors = [], [], []
    used_labels = set()
    for image in images:
        key = image.relative_to(image_root).with_suffix("").as_posix().casefold()
        candidates = by_path.get(key, [])
        if not candidates:
            same_name = by_stem.get(image.stem.casefold(), [])
            if not same_name:
                # 模糊图等未标注图片：跳过，不自动生成空标签。
                skipped.append(image)
                continue
            if image_stem_counts[image.stem.casefold()] != 1:
                # 图片跨批次重名时，仅认可相同相对路径下的标签。
                # 此处没有对应路径的标签，按未标注跳过，不能借用其他批次的标签。
                skipped.append(image)
                continue
            if len(same_name) != 1:
                errors.append(f"同名文件无法确定配对：{image}；请让图片和标签子目录结构一致。")
                continue
            candidates = same_name
        if len(candidates) != 1:
            errors.append(f"存在多个对应标签：{image}")
        elif candidates[0] in used_labels:
            errors.append(f"同一个标签对应多张图片：{candidates[0]}")
        else:
            pairs.append((image, candidates[0]))
            used_labels.add(candidates[0])

    print(f"找到图片 {len(images)} 张，可整合 {len(pairs)} 对，未标注跳过 {len(skipped)} 张。")
    for image in skipped[:10]:
        print(f"跳过：{image}")
    if len(skipped) > 10:
        print(f"其余 {len(skipped) - 10} 张跳过图片不逐一显示。")
    if errors:
        raise ValueError(
            f"存在 {len(errors)} 个配对问题，未创建输出目录：\n" + "\n".join(errors[:30])
        )
    if not pairs:
        raise ValueError("没有可整合的图片和标签，未创建输出目录。请检查标签路径和文件名。")

    # 创建新目录，复制源文件；不移动或删除原图片和标签。
    output_root.mkdir(parents=True, exist_ok=False)
    picture_dir, label_dir = output_root / "picture", output_root / "label"
    picture_dir.mkdir()
    label_dir.mkdir()
    width = max(6, len(str(len(pairs))))
    for index, (image, label) in enumerate(pairs, start=1):
        # 两者使用相同编号，消除不同批次的重名；标签内容原样保留。
        # 已存在的空 txt 标签也保留，它与没有 txt 文件是两种情况。
        stem = f"{index:0{width}d}"
        shutil.copy2(image, picture_dir / f"{stem}{image.suffix.lower()}")
        shutil.copy2(label, label_dir / f"{stem}.txt")
        if index % 500 == 0 or index == len(pairs):
            print(f"复制进度：{index}/{len(pairs)}")

    print(f"\n整合成功：{len(pairs)} 对图片和标签，跳过 {len(skipped)} 张未标注图片。")
    print(f"输出文件夹：{output_root}")
    print(f"图片：{picture_dir}\n标签：{label_dir}")
    print(f"未使用的 txt 文件数量：{len(labels) - len(used_labels)}（包括可能存在的类别说明文件）。")
    return output_root


if __name__ == "__main__":
    # 各批次的类别编号含义必须一致，本脚本不修改类别编号。
    try:
        merge_dataset(IMAGE_ROOT, LABEL_ROOT, OUTPUT_PARENT, OUTPUT_NAME)
    except (ValueError, OSError) as exc:
        print(f"\n整合失败：{exc}")
        print("若发生复制错误，输出可能不完整，请修复问题并更换 OUTPUT_NAME 后重试。")
        raise SystemExit(1)
