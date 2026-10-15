"""既有电控协议：保持包头、包尾、颜色编号、float32小端和任务确认规则。"""
import math
import queue
import re
import struct
import threading
import time

TASK_PATTERN = re.compile(rb"[1-6]{3}\+[1-6]{3}\+[1-6]{3}\+[1-6]{3}")
COLOR_IDS = {"red": 1, "yellow": 2, "darkblue": 3, "green": 4, "black": 5, "lightblue": 6}


def float_mm(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("禁止发送非有限坐标")
    return struct.pack("<f", round(value, 1))


def material_packet(color, x, y, z):
    """17字节；通信编号与YOLO类别编号不同，必须经COLOR_IDS转换。"""
    return b"\xaa\xbb" + bytes([COLOR_IDS[color]]) + b"".join(map(float_mm, (x, y, z))) + b"\xcc\xdd"


def target_packet(number, x, y):
    """13字节。保持原协议，圆环高度只显示，不向旧数据包追加字段。"""
    if number not in (1, 2, 3):
        raise ValueError("圆环编号必须是1/2/3")
    return b"\xbb\xaa" + bytes([number]) + float_mm(x) + float_mm(y) + b"\xdd\xcc"


class TaskParser:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        events = []
        while self.buffer:
            start = self.buffer.find(b"CSCS")
            if start < 0:
                keep = max((n for n in range(1, 4) if self.buffer.endswith(b"CSCS"[:n])), default=0)
                self.buffer[:] = self.buffer[-keep:] if keep else b""
                break
            del self.buffer[:start]
            if len(self.buffer) < 23:
                break
            payload = bytes(self.buffer[4:19])
            if self.buffer[19:23] == b"GOGO" and TASK_PATTERN.fullmatch(payload):
                events.append(payload.decode("ascii"))
                del self.buffer[:23]
            else:
                events.append(None)
                del self.buffer[:1]
        return events


class TaskConfirmation:
    def __init__(self):
        self.candidate = None
        self.count = 0
        self.locked_code = None

    def accept(self, code):
        if self.locked_code is not None:
            return None
        if code is None:
            self.candidate, self.count = None, 0
            return None
        self.count = self.count + 1 if code == self.candidate else 1
        self.candidate = code
        if self.count >= 2:
            self.locked_code = code
            return code
        return None


class AckBurst:
    def __init__(self):
        self.start_time = None
        self.next_time = 0.0

    def due(self, now, requested):
        if requested and self.start_time is None:
            self.start_time = self.next_time = now
        if self.start_time is None or now >= self.start_time + 2.0 or now < self.next_time:
            return False
        self.next_time = now + 0.05
        return True


class SerialLink:
    """一个线程独占串口；坐标按目标替换排队，超时丢弃，防止发送积压旧位置。"""
    def __init__(self, port, baudrate=115200, enabled=True, serial_factory=None):
        self.port, self.baudrate, self.enabled = port, baudrate, enabled
        self.factory = serial_factory
        self.confirmation = TaskConfirmation()
        self.parser = TaskParser()
        self.confirmed = queue.Queue()
        self.pending = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ack_request = threading.Event()
        self.available = False
        self.status = "UART OFF" if not enabled else "UART STARTING"
        self.thread = None

    def start(self):
        if self.enabled:
            self.thread = threading.Thread(target=self._run, daemon=True, name="uart")
            self.thread.start()

    def submit_batch(self, group, packets, expiry):
        if self.available:
            with self.lock:
                self.pending[group] = (list(packets), expiry)

    def acknowledge_display(self, code):
        if code and code == self.confirmation.locked_code:
            self.ack_request.set()

    def _run(self):
        port = None
        try:
            factory = self.factory
            if factory is None:
                from serial import Serial
                factory = Serial
            port = factory(self.port, self.baudrate, bytesize=8, parity="N", stopbits=1,
                           timeout=0, write_timeout=0.2)
            self.available = True
            self.status = "UART WAITING 0/2"
            ack = AckBurst()

            def write_all(data):
                sent = 0
                while sent < len(data):
                    n = port.write(data[sent:])
                    if not n:
                        raise IOError("串口数据包未写完整")
                    sent += n

            while not self.stop.is_set():
                if not port.is_open:
                    raise IOError("串口已断开")
                if port.in_waiting:
                    for code in self.parser.feed(port.read(min(port.in_waiting, 4096))):
                        confirmed = self.confirmation.accept(code)
                        self.status = "UART LOCKED" if self.confirmation.locked_code else f"UART WAITING {self.confirmation.count}/2"
                        if confirmed:
                            print("[UART] 任务码锁定：", confirmed, flush=True)
                            self.confirmed.put(confirmed)
                if ack.due(time.monotonic(), self.ack_request.is_set()):
                    write_all(b"F")
                with self.lock:
                    pending, self.pending = self.pending, {}
                for packets, expiry in pending.values():
                    for packet in packets:
                        if time.monotonic() < expiry:
                            write_all(packet)
                self.stop.wait(0.005)
        except Exception as exc:
            self.status = "UART ERROR - CHECK CONSOLE"
            print(f"[UART] {self.port}: {exc}", flush=True)
        finally:
            self.available = False
            if port is not None:
                port.close()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(1.0)
