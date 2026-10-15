"""采集与推理分离：只处理最新帧，检测结果始终携带其对应原图和时间戳。"""
import os
from pathlib import Path
import threading
import time

import cv2

from geometry import CameraGeometry, measure


class CameraStream:
    def __init__(self, cfg, source=None):
        self.cfg = cfg
        self.source = source if source is not None else cfg["windows_source" if os.name == "nt" else "linux_source"]
        if isinstance(self.source,str) and self.source.isdigit():
            self.source = int(self.source)
        self.lock = threading.Lock()
        self.latest = None
        self.stop = threading.Event()
        self.error = None
        self.info = "CAMERA STARTING"
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera")

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.latest

    def _run(self):
        cap = None
        try:
            # Windows 与摄像头预览脚本保持一致，交给 OpenCV 选择后端。
            # 某些 UVC 驱动在 DirectShow 下只返回 YUY2，而自动后端可协商 MJPG。
            live = isinstance(self.source,int) or str(self.source).startswith("/dev/")
            backend = cv2.CAP_V4L2 if live and os.name != "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.source, backend)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开摄像头/视频：{self.source}")
            if live:
                # 顺序与相机示例一致：先MJPG，再分辨率，再帧率。
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])
                cap.set(cv2.CAP_PROP_FPS, self.cfg["fps"])
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            code = int(cap.get(cv2.CAP_PROP_FOURCC))
            fmt = "".join(chr((code >> 8*i) & 255) for i in range(4))
            width,height = int(cap.get(3)),int(cap.get(4))
            self.info = f"{fmt} {width}x{height} {cap.get(5):.1f}FPS"
            print("[CAMERA]", self.info, flush=True)
            if live and self.cfg["require_mjpg"] and fmt != "MJPG":
                raise RuntimeError(f"摄像头未采用MJPG，实际为{fmt}；请检查驱动和支持的分辨率")
            seq = 0
            while not self.stop.is_set():
                ok,frame = cap.read()
                if not ok:
                    raise RuntimeError("摄像头读取失败或视频已结束")
                seq += 1
                with self.lock:
                    self.latest = (seq, time.monotonic(), frame)
                if not live:
                    self.stop.wait(1/max(cap.get(cv2.CAP_PROP_FPS),1))
        except Exception as exc:
            self.error = str(exc)
            print("[CAMERA]",self.error,flush=True)
        finally:
            if cap is not None:
                cap.release()

    def close(self):
        self.stop.set()
        self.thread.join(2)


class Detector:
    """模型实例只被推理线程使用；新摄像头可另建实例，不共享帧或测量状态。"""
    def __init__(self, cfg, base):
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import DEFAULT_CFG_DICT
        self.cfg = cfg
        self.camera = CameraGeometry(cfg["geometry"],base)
        path = (Path(base)/cfg["model"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到本地模型：{path}；请修改config.json，不会自动下载模型")
        torch.set_num_threads(cfg["inference"]["cpu_threads"])
        self.model = YOLO(str(path))
        if self.model.task != "detect":
            raise ValueError("当前程序需要矩形框目标检测权重")
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        expected = cfg["class_names"]
        if self.names != expected:
            raise ValueError(f"权重类别顺序与配置不同！权重={self.names}，配置={expected}；先核对标注，不自动重命名")
        supported = {"red","yellow","green","black","lightblue","darkblue","target","target1","target2","target3"}
        if not set(self.names) <= supported:
            raise ValueError("模型含未支持类别；请先配置类别到测量任务的映射")
        infer = cfg["inference"]
        device = infer["device"]
        if device == "auto":
            device = 0 if torch.cuda.is_available() else "cpu"
        self.args = dict(device=device, imgsz=infer["imgsz"], conf=infer["conf"], iou=infer["iou"],
                         max_det=infer["max_det"], agnostic_nms=infer["agnostic_nms"], verbose=False)
        if infer["half"] and device != "cpu":
            self.args.update({"quantize":16} if "quantize" in DEFAULT_CFG_DICT else {"half":True})
        print(f"[MODEL] {path}\n[CLASSES] {self.names}\n[DEVICE] {device}",flush=True)

    def process(self, frame):
        frame = self.camera.prepare(frame)
        result = self.model.predict(frame, **self.args)[0]
        items = []
        if result.boxes is not None:
            for row in result.boxes.data.cpu().numpy():
                x1,y1,x2,y2,confidence,class_id = row[:6]
                name = self.model.names[int(class_id)]
                items.append(measure(frame, (x1,y1,x2,y2), name, float(confidence), self.camera))
        # 圆环印刷黑线不再被当作黑色物料发送；只对已通过几何验证的target排除。
        rings = [m for m in items if m["valid"] and m.get("mode") == "INNER EDGE"]
        for item in items:
            if item["name"] != "black" or not item["valid"]:
                continue
            cx,cy = item["center"]
            if any(r["box"][0] <= cx <= r["box"][2] and r["box"][1] <= cy <= r["box"][3] for r in rings):
                item.update(valid=False,reason="OVERLAPS TARGET")
        return frame, items


class InferenceWorker:
    def __init__(self, stream, cfg, base):
        self.stream,self.cfg,self.base = stream,cfg,base
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.status = "MODEL LOADING"
        self.error = None
        self.thread = threading.Thread(target=self._run,daemon=True,name="inference")

    def start(self):
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return self.latest

    def _run(self):
        try:
            detector = Detector(self.cfg,self.base)
            self.status = "MODEL READY"
            last = -1
            while not self.stop.is_set():
                sample = self.stream.snapshot()
                if sample is None or sample[0] == last:
                    self.stop.wait(.005)
                    continue
                seq,stamp,frame = sample
                last = seq
                start = time.monotonic()
                frame,items = detector.process(frame)
                elapsed = time.monotonic()-start
                with self.lock:
                    self.latest = (seq,stamp,frame,items,elapsed)
                self.status = f"INFER {elapsed*1000:.0f}ms"
        except Exception as exc:
            self.error = str(exc)
            self.status = "MODEL ERROR - CHECK CONSOLE"
            import traceback
            traceback.print_exc()

    def close(self):
        self.stop.set()
        self.thread.join(3)
