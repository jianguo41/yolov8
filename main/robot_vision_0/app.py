"""独立入口。双击启动器；或 python app.py --help。."""

import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

# 先启动串口，再导入OpenCV/torch。所有路径基于本文件，支持整个目录迁移。
from protocol import TASK_PATTERN, SerialLink, TaskConfirmation, material_packet, target_packet

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE / ".runtime"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["task_code"]["source"] not in ("serial", "camera"):
        raise ValueError("task_code.source 必须是 serial 或 camera")
    for key in (
        "material_top_diameter_mm",
        "material_bottom_diameter_mm",
        "material_height_mm",
        "target_line_width_mm",
    ):
        if cfg["geometry"][key] <= 0:
            raise ValueError(f"{key} 必须大于0")
    return cfg, path.parent


class QRWorker:
    """可选相机扫码，与串口接收模式互斥；二维码连续两次相同才锁定。."""

    def __init__(self, stream, interval):
        self.stream, self.interval = stream, interval
        self.confirmation = TaskConfirmation()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True, name="qr")

    def run(self):
        import cv2

        detector = cv2.QRCodeDetector()
        last = -1
        try:
            while not self.stop.wait(self.interval):
                sample = self.stream.snapshot()
                if sample is None or sample[0] == last or self.confirmation.locked_code:
                    continue
                last = sample[0]
                value, _, _ = detector.detectAndDecode(sample[2])
                data = value.encode("ascii", errors="ignore")
                self.confirmation.accept(value if TASK_PATTERN.fullmatch(data) and len(value) == 15 else None)
        except Exception as exc:
            print("[QR]", exc, flush=True)

    def close(self):
        self.stop.set()
        self.thread.join(1)


def offline(args, cfg, base):
    import cv2
    import numpy as np
    from display import annotate, draw_task, letterbox, measurement_table, text
    from runtime import Detector

    detector = Detector(cfg, base)
    if args.check:
        print("模型与配置检查通过。标定状态：", detector.camera.calibrated)
        return
    source = Path(args.image)
    frame = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"无法读取图片：{source}")
    start = time.monotonic()
    frame, items = detector.process(frame)
    elapsed = time.monotonic() - start
    canvas = letterbox(annotate(frame, items), cfg["display"])
    draw_task(canvas, args.task_code, cfg["display"])
    measurement_table(canvas, items)
    text(
        canvas,
        f"OFFLINE {elapsed * 1000:.0f}ms / "
        + ("CALIBRATED" if detector.camera.calibrated else "UNCALIBRATED ESTIMATES"),
        (12, cfg["display"]["height"] - 15),
        0.6,
    )
    output = Path(args.output or str(base / "output" / "preview.jpg"))
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output.suffix, canvas)
    if not ok:
        raise RuntimeError("输出图像编码失败")
    encoded.tofile(str(output))
    output.with_suffix(".json").write_text(
        json.dumps({"seconds": elapsed, "items": items}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("输出：", output, "检测：", len(items), "有效几何测量：", sum(i["valid"] for i in items))


def live(args, cfg, base):
    serial_cfg = cfg["serial"]
    port = args.port or serial_cfg["windows_port" if os.name == "nt" else "linux_port"]
    link = SerialLink(port, serial_cfg["baudrate"], serial_cfg["enabled"] and not args.no_serial)
    link.start()
    stream = worker = qr = None
    try:
        import cv2
        import numpy as np
        from display import annotate, draw_task, letterbox, measurement_table, task_layout, text
        from runtime import CameraStream, InferenceWorker

        display = cfg["display"]
        task_layout("123+123+123+123", display)  # 启动时确认字高与布局能够同时满足。
        name = display["window_name"]
        if not args.headless:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            if display["fullscreen"]:
                cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(name, display["width"], display["height"])
        stream = CameraStream(cfg["camera"], args.source)
        stream.start()
        worker = InferenceWorker(stream, cfg, base)
        worker.start()
        if cfg["task_code"]["source"] == "camera":
            qr = QRWorker(stream, cfg["task_code"]["camera_scan_interval_s"])
            qr.thread.start()
        locked_code = None
        shown_seq = sent_seq = -1
        observed = None
        rings_time = materials_time = 0
        loops = 0
        while True:
            now = time.monotonic()
            if locked_code is None:
                if qr:
                    locked_code = qr.confirmation.locked_code
                else:
                    try:
                        locked_code = link.confirmed.get_nowait()
                    except queue.Empty:
                        pass
            result = worker.snapshot()
            raw = stream.snapshot()
            age = now - result[1] if result else float("inf")
            fresh = (
                result is not None and age <= serial_cfg["max_result_age_s"] and not stream.error and not worker.error
            )
            if fresh:
                seq, _stamp, frame, items, _elapsed = result
                if shown_seq != seq:
                    observed = annotate(frame, items)
                    shown_seq = seq
            else:
                # 不把旧框叠加在最新画面上；结果过期时仅显示实时预览。
                observed = (
                    raw[2].copy() if raw else np.zeros((cfg["camera"]["height"], cfg["camera"]["width"], 3), np.uint8)
                )
                shown_seq = -1
            if fresh and sent_seq != result[0]:
                sent_seq = result[0]
                valid = [
                    m for m in result[3] if m["valid"] and (m["calibrated"] or not serial_cfg["require_calibration"])
                ]
                expiry = result[1] + serial_cfg["max_result_age_s"]
                if now - rings_time >= serial_cfg["ring_interval_s"]:
                    packets = [
                        target_packet(m["number"], m["x"], m["y"])
                        for m in valid
                        if m["name"].startswith("target") and m.get("number")
                    ]
                    link.submit_batch("targets", packets, expiry)
                    rings_time = now
                if now - materials_time >= serial_cfg["material_interval_s"]:
                    packets = [
                        material_packet(m["name"], m["x"], m["y"], m["z"])
                        for m in valid
                        if not m["name"].startswith("target")
                    ]
                    link.submit_batch("materials", packets, expiry)
                    materials_time = now
                    for m in result[3]:
                        if m["valid"]:
                            print(
                                f"[MEASURE] {m['name']} X={m['x']:+.1f} Y={m['y']:+.1f} D={m['distance']:.1f} Z={m['z']:.1f} mm"
                                + (
                                    " [ESTIMATE: " + m["mode"] + "]"
                                    if m.get("estimated")
                                    else ""
                                    if m["calibrated"]
                                    else " [ESTIMATE / UNCALIBRATED]"
                                ),
                                flush=True,
                            )
            if not args.headless:
                canvas = letterbox(observed, display)
                draw_task(canvas, locked_code, display)
                if fresh:
                    measurement_table(canvas, result[3])
                status = f"{worker.status} | {link.status} | " + ("MEASURE" if fresh else "PREVIEW / WAITING OR STALE")
                text(canvas, status, (12, display["height"] - 38), 0.53, (0, 255, 255))
                calibration = base / cfg["geometry"]["calibration"]
                text(
                    canvas,
                    "X right  Y up  Z camera-to-plane | "
                    + (
                        "CALIBRATION LOADED"
                        if calibration.exists()
                        else "UNCALIBRATED - ESTIMATES / METRIC TX OFF"
                        if serial_cfg["require_calibration"]
                        else "UNCALIBRATED - ESTIMATED TX"
                    ),
                    (12, display["height"] - 13),
                    0.5,
                )
                cv2.imshow(name, canvas)
                key = cv2.waitKey(1) & 255
                # 保持显示完成后才回复F；相机扫码模式不冒充串口双帧确认。
                if locked_code and not qr:
                    link.acknowledge_display(locked_code)
                if key in (27, ord("q")) or cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            else:
                time.sleep(0.01)
            if stream.error or worker.error:
                raise RuntimeError(stream.error or worker.error)
            loops += 1
            if args.max_frames and loops >= args.max_frames:
                break
    finally:
        if qr:
            qr.close()
        if worker:
            worker.close()
        if stream:
            stream.close()
        link.close()
        if "cv2" in sys.modules:
            sys.modules["cv2"].destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(BASE / "config.json"))
    parser.add_argument("--source", help="覆盖摄像头编号、/dev/camera1或测试视频路径")
    parser.add_argument("--model", help="覆盖本地权重路径")
    parser.add_argument("--port", help="覆盖串口路径")
    parser.add_argument("--no-serial", action="store_true")
    parser.add_argument("--headless", action="store_true", help="无窗口；不会对未显示的任务码回复F")
    parser.add_argument("--image", help="离线图片测试，不打开摄像头或串口")
    parser.add_argument("--output", help="离线结果图片路径，同时保存JSON")
    parser.add_argument("--task-code", help="仅离线界面预览用，如156+123+516+231")
    parser.add_argument("--check", action="store_true", help="仅检查权重和类别配置")
    parser.add_argument("--max-frames", type=int, default=0, help="限制预览循环次数，用于启动测试")
    args = parser.parse_args()
    if args.task_code and not TASK_PATTERN.fullmatch(args.task_code.encode("ascii")):
        parser.error("任务码格式错误")
    cfg, base = read_config(args.config)
    if args.model:
        cfg["model"] = str(Path(args.model).resolve())
    if args.image or args.check:
        offline(args, cfg, base)
    else:
        live(args, cfg, base)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("已停止")
