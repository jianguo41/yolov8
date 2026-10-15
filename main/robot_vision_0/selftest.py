"""离线校验：不打开摄像头、不连接真实串口、不修改权重和标定。."""

import json
import struct
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from display import draw_task, task_layout
from geometry import CameraGeometry, measure
from protocol import AckBurst, SerialLink, TaskConfirmation, TaskParser, material_packet, target_packet

BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / "config.json").read_text(encoding="utf-8"))


def camera():
    model = CameraGeometry({**CFG["geometry"], "calibration": "does_not_exist.json"}, BASE)
    model.calibration = {"kind": "scale", "image_size": [1280, 720], "focal_width_ratio": 1000 / 1280}
    model.prepare(np.zeros((720, 1280, 3), np.uint8))
    return model


class Checks(unittest.TestCase):
    def test_serial_worker_partial_write_and_expiry(self):
        class Port:
            def __init__(self, *args, **kwargs):
                self.input = bytearray(b"CSCS156+123+516+231GOGO" * 2)
                self.output = bytearray()
                self.is_open = True

            @property
            def in_waiting(self):
                return len(self.input)

            def read(self, n):
                value = bytes(self.input[:n])
                del self.input[:n]
                return value

            def write(self, data):
                self.output.extend(data[:2])
                return min(len(data), 2)

            def close(self):
                self.is_open = False

        port = Port()
        link = SerialLink("TEST", serial_factory=lambda *a, **k: port)
        link.start()
        try:
            code = link.confirmed.get(timeout=1)
            self.assertEqual(code, "156+123+516+231")
            self.assertEqual(port.output, b"")  # 尚未显示，不能发F。
            link.submit_batch("expired", [b"STALE"], time.monotonic() - 1)
            packet = material_packet("red", 1, 2, 3)
            link.submit_batch("materials", [packet], time.monotonic() + 1)
            link.acknowledge_display(code)
            deadline = time.monotonic() + 1
            while packet not in port.output and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIn(packet, port.output)  # 即使短写，F也不能插进坐标包中。
            self.assertIn(b"F", port.output)
            self.assertNotIn(b"STALE", port.output)
        finally:
            link.close()
        self.assertFalse(port.is_open)

    def test_wire_bytes(self):
        packet = material_packet("lightblue", 12.34, -8.76, 456.78)
        self.assertEqual(packet[:3], b"\xaa\xbb\x06")
        self.assertEqual(packet[-2:], b"\xcc\xdd")
        self.assertEqual(len(packet), 17)
        for value, expected in zip(struct.unpack("<fff", packet[3:-2]), (12.3, -8.8, 456.8)):
            self.assertAlmostEqual(value, expected, places=3)
        self.assertEqual(target_packet(2, 1, -2), b"\xbb\xaa\x02" + struct.pack("<ff", 1, -2) + b"\xdd\xcc")
        with self.assertRaises(ValueError):
            material_packet("red", float("nan"), 0, 1)

    def test_split_noise_and_corrupt_frames(self):
        frame = b"CSCS156+123+516+231GOGO"
        for split in range(1, len(frame)):
            parser = TaskParser()
            self.assertEqual(parser.feed(b"noise" + frame[:split]) + parser.feed(frame[split:]), ["156+123+516+231"])
        parser = TaskParser()
        events = parser.feed(b"CSCS000+000+000+000GOGO" + frame + frame)
        self.assertEqual(events, [None, "156+123+516+231", "156+123+516+231"])

    def test_confirmation_and_ack(self):
        confirm = TaskConfirmation()
        for code in ("A", None, "B"):
            self.assertIsNone(confirm.accept(code))
        self.assertEqual(confirm.accept("B"), "B")
        self.assertIsNone(confirm.accept("C"))
        self.assertEqual(confirm.locked_code, "B")
        ack = AckBurst()
        self.assertFalse(ack.due(0, False))
        self.assertTrue(ack.due(1, True))
        self.assertFalse(ack.due(1.01, True))
        self.assertTrue(ack.due(1.06, True))
        self.assertFalse(ack.due(3.0, True))
        self.assertFalse(ack.due(4.0, True))

    def test_bottom_projection(self):
        model = camera()
        center, z = model.bottom_from_top((740, 310), 400)
        self.assertEqual(z, 460)
        x, y, d = model.position(center, z)
        self.assertAlmostEqual(x, 40)
        self.assertAlmostEqual(y, 20)
        self.assertAlmostEqual(d, np.hypot(40, 20))

    def test_target_inner_edge(self):
        frame = np.full((720, 1280, 3), 255, np.uint8)
        # f=1000,Z=500时，圆的像素半径数值恰等于直径毫米数值。
        for outer in reversed((53, 58, 65, 75, 85, 95)):
            cv2.circle(frame, (700, 300), outer, (0, 0, 0), -1)
            cv2.circle(frame, (700, 300), outer - 3, (255, 255, 255), -1)
        item = measure(frame, (597, 197, 803, 403), "target2", 0.95, camera())
        self.assertTrue(item["valid"], item)
        self.assertAlmostEqual(item["z"], 500, delta=15)
        self.assertAlmostEqual(item["x"], 30, delta=2)
        self.assertAlmostEqual(item["y"], 30, delta=2)

    def test_background_and_clipping(self):
        frame = np.full((720, 1280, 3), 255, np.uint8)
        target = measure(frame, (100, 100, 300, 300), "target1", 0.9, camera())
        self.assertTrue(target["valid"])
        self.assertEqual(target["mode"], "BOX EST")
        self.assertEqual(target["center"], (200, 200))
        self.assertEqual(len(target_packet(target["number"], target["x"], target["y"])), 13)
        clipped = measure(frame, (0, 100, 200, 300), "red", 0.9, camera())
        self.assertTrue(clipped["valid"])
        self.assertEqual(clipped["center"], (100, 200))
        self.assertEqual(len(material_packet("red", clipped["x"], clipped["y"], clipped["z"])), 17)
        self.assertFalse(measure(frame, (100, 100, 100, 300), "red", 0.9, camera())["valid"])

    def test_bottom_outline_without_top(self):
        frame = np.full((720, 1280, 3), 255, np.uint8)
        cv2.circle(frame, (700, 300), 50, (65, 21, 253), -1)
        item = measure(frame, (646, 246, 754, 354), "red", 0.95, camera())
        self.assertTrue(item["valid"], item)
        self.assertEqual(item["mode"], "BOTTOM OUTLINE")
        self.assertAlmostEqual(item["center"][0], 700, delta=3)
        self.assertAlmostEqual(item["z"], 500, delta=25)

    def test_material_top_and_bottom(self):
        frame = np.full((720, 1280, 3), 255, np.uint8)
        cv2.circle(frame, (640, 360), 54, (76, 59, 198), -1)
        cv2.circle(frame, (648, 352), 32, (65, 21, 253), -1)
        item = measure(frame, (581, 301, 699, 419), "red", 0.95, camera())
        self.assertTrue(item["valid"], item)
        self.assertEqual(item["mode"], "TOP CIRCLE")
        self.assertIn("top_depth", item)
        self.assertAlmostEqual(item["z"], 60 + 30 * 1000 / 64, delta=25)
        self.assertAlmostEqual(item["display_center"][0], 648, delta=3)
        self.assertAlmostEqual(item["display_center"][1], 352, delta=3)
        self.assertIsNone(item["ellipse"])
        self.assertAlmostEqual(item["top_circle"][0], item["display_center"][0], delta=0.1)

    def test_task_display(self):
        cfg = CFG["display"]
        _, _, sizes, _, _ = task_layout("156+123+516+231", cfg)
        self.assertTrue(
            all(s[0][1] * cfg["physical_height_mm"] / cfg["height"] >= cfg["task_font_height_mm"] for s in sizes)
        )
        canvas = np.zeros((cfg["height"], cfg["width"], 3), np.uint8)
        draw_task(canvas, "156+123+516+231", cfg)
        self.assertGreater(np.count_nonzero(canvas), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
