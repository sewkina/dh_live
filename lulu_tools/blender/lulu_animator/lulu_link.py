"""噜噜动作编辑器的连接层和算法，不依赖 Blender，Houdini 或命令行也能直接用。

- 舵机换算：关节角度（度）<-> 舵机偏移（相对零点，与固件 set_motor_pos 一致）
- 连接：USB 串口（pyserial）或蓝牙（bleak），协议见固件 kf_link.h
- 动作 JSON 读写、去抖、RDP 精简
- 简单运动学：脚的位置、翻倒和腿互撞检查

命令行测试（不开 Blender）：
    python lulu_link.py ports
    python lulu_link.py ping COM3
    python lulu_link.py play wave.json COM3
    python lulu_link.py install wave.json COM3
"""

import json
import math
import queue
import sys
import threading
import time

# ---------------------------------------------------------------- 关节定义

# 顺序与固件舵机顺序一致：左前、右前、左后、右后、腰
JOINTS = ["leg_LF", "leg_RF", "leg_LB", "leg_RB", "waist"]
JOINT_LABELS = ["左前腿", "右前腿", "左后腿", "右后腿", "腰"]
NUM_JOINTS = 5

# 角度约定与 MCP 工具 self.dog.set_motor_angle 相同：
# 腿 0 度垂直向下，90 度水平（趴下），站立约 36 度；腰正值上半身向左转
STAND_ANGLES = [36.0, 36.0, 36.0, 36.0, 0.0]
ANGLE_MIN = [-20.0, -20.0, -20.0, -20.0, -40.0]
ANGLE_MAX = [135.0, 135.0, 135.0, 135.0, 40.0]

_UNITS_PER_DEG = 3000.0 / 270.0  # EM3 舵机：3000 单位 = 270 度


def angle_to_offset(i, angle):
    """关节角度（度）-> 舵机偏移，公式来自固件 set_motor_angle。"""
    if i in (0, 2):
        return int(round(angle * _UNITS_PER_DEG - 1000))
    if i in (1, 3):
        return int(round(-angle * _UNITS_PER_DEG + 1000))
    return int(round(angle * _UNITS_PER_DEG))


def offset_to_angle(i, offset):
    if i in (0, 2):
        return (offset + 1000) / _UNITS_PER_DEG
    if i in (1, 3):
        return (1000 - offset) / _UNITS_PER_DEG
    return offset / _UNITS_PER_DEG


def clamp_angles(angles):
    return [min(max(a, ANGLE_MIN[i]), ANGLE_MAX[i]) for i, a in enumerate(angles)]


# ---------------------------------------------------------------- 动作 JSON

def make_action(name, frames, interp="spline", emotion=None, loop=None):
    """frames: [(d_ms, [5 个角度]), ...] -> 固件动作 dict。"""
    action = {"name": name}
    if emotion:
        action["emotion"] = emotion
    if interp == "spline":
        action["interp"] = "spline"
    action["frames"] = [
        {"d": int(max(20, min(10000, round(d)))),
         "p": [angle_to_offset(i, a) for i, a in enumerate(clamp_angles(angles))]}
        for d, angles in frames
    ]
    if loop:
        action["loop"] = loop
    return action


def action_frames(action):
    """固件动作 dict -> [(d_ms, [5 个角度]), ...]"""
    return [(f["d"], [offset_to_angle(i, p) for i, p in enumerate(f["p"])]) for f in action["frames"]]


def dumps_action(action):
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


def validate_action(action):
    problems = []
    if not action.get("name"):
        problems.append("缺少 name")
    frames = action.get("frames", [])
    if not 1 <= len(frames) <= 40:
        problems.append("帧数需在 1~40 之间（现在 %d）" % len(frames))
    if len(dumps_action(action).encode("utf-8")) >= 4000:
        problems.append("JSON 超过 4000 字节，噜噜存不下，请减少帧数")
    return problems


# ---------------------------------------------------------------- 去抖和精简

def smooth(poses, window=3):
    """滑动平均去掉手抖。poses: [[5 个角度], ...]"""
    if window <= 1 or len(poses) < 3:
        return [list(p) for p in poses]
    half = window // 2
    out = []
    for k in range(len(poses)):
        lo, hi = max(0, k - half), min(len(poses), k + half + 1)
        out.append([sum(poses[j][i] for j in range(lo, hi)) / (hi - lo) for i in range(NUM_JOINTS)])
    return out


def rdp_indices(poses, epsilon):
    """Ramer-Douglas-Peucker：保留转折点，返回保留的下标。误差按各关节角度最大偏差（度）算。"""
    n = len(poses)
    if n <= 2:
        return list(range(n))
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        worst, worst_k = 0.0, -1
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            err = max(abs(poses[k][i] - (poses[a][i] + (poses[b][i] - poses[a][i]) * t)) for i in range(NUM_JOINTS))
            if err > worst:
                worst, worst_k = err, k
        if worst > epsilon and worst_k > 0:
            keep.add(worst_k)
            stack.append((a, worst_k))
            stack.append((worst_k, b))
    return sorted(keep)


def samples_to_frames(times_ms, poses, epsilon=1.5, max_frames=40, smooth_window=3):
    """连续采样 -> 关键帧 [(d_ms, angles)]。第一帧的 d 是从当前姿态过渡过去的时间。"""
    poses = smooth(poses, smooth_window)
    eps = epsilon
    while True:
        idx = rdp_indices(poses, eps)
        if len(idx) <= max_frames:
            break
        eps *= 1.3
    frames = []
    prev_t = times_ms[idx[0]]
    for n, k in enumerate(idx):
        d = 600 if n == 0 else times_ms[k] - prev_t
        frames.append((d, poses[k]))
        prev_t = times_ms[k]
    return frames, eps


# ---------------------------------------------------------------- 运动学和检查

# 尺寸（米），按官网照片估算，误差约 2~3mm，待卡尺实测
DIMS = {
    "hip_dx": 0.032,      # 髋关节到中线的横向距离
    "hip_dy": 0.030,      # 髋关节到腰关节的前后距离
    "leg_len": 0.045,     # 髋到脚尖
    "leg_thick": 0.009,   # 腿宽
    "front_mass": 0.6,    # 前半身（含头和屏幕）占总重的比例，估算
    "front_com_dy": 0.030,  # 前半身重心到腰关节的距离
    "rear_com_dy": 0.024,   # 后半身重心到腰关节的距离
}


def foot_positions(angles, dims=DIMS):
    """以后半身为参考、腰关节为原点：x 向右，y 向前，z 向上。返回 [(hip, foot), ...]，顺序同 JOINTS 前 4 个。"""
    yaw = math.radians(angles[4])
    cy, sy = math.cos(yaw), math.sin(yaw)
    out = []
    for i in range(4):
        side = -1.0 if i in (0, 2) else 1.0   # 左腿在 -x
        front = i in (0, 1)
        a = math.radians(angles[i])
        hx, hy = side * dims["hip_dx"], dims["hip_dy"] if front else -dims["hip_dy"]
        fy = math.sin(a) * dims["leg_len"] * (1 if front else -1)
        fz = -math.cos(a) * dims["leg_len"]
        hip = (hx, hy, 0.0)
        foot = (hx, hy + fy, fz)
        if front:  # 前半身随腰转（绕 z 轴，正值向左）
            hip = (hip[0] * cy - hip[1] * sy, hip[0] * sy + hip[1] * cy, 0.0)
            foot = (foot[0] * cy - foot[1] * sy, foot[0] * sy + foot[1] * cy, fz)
        out.append((hip, foot))
    return out


def _seg_dist_2d(p1, p2, q1, q2):
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1])

    def point_seg(p, a, b):
        ab, ap = sub(b, a), sub(p, a)
        t = max(0.0, min(1.0, dot(ap, ab) / max(dot(ab, ab), 1e-12)))
        c = (a[0] + ab[0] * t, a[1] + ab[1] * t)
        return math.hypot(p[0] - c[0], p[1] - c[1])

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    if (ccw(p1, q1, q2) > 0) != (ccw(p2, q1, q2) > 0) and (ccw(p1, p2, q1) > 0) != (ccw(p1, p2, q2) > 0):
        return 0.0
    return min(point_seg(p1, q1, q2), point_seg(p2, q1, q2), point_seg(q1, p1, p2), point_seg(q2, p1, p2))


def _support_margin(pt, poly):
    """点到凸多边形边界的有符号距离：正数在里面，负数在外面（米）。"""
    if len(poly) < 3:
        return -1.0
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    poly = sorted(poly, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    margin = float("inf")
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        length = math.hypot(b[0] - a[0], b[1] - a[1]) or 1e-9
        margin = min(margin, ((b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])) / length)
    return margin


def check_pose(angles, dims=DIMS, contact_tol=0.004, tip_margin=0.025):
    """返回问题列表（空表示没问题）：翻倒风险、同侧前后腿相撞、超出关节限位。
    只是粗略估算，最终以真机为准。"""
    problems = []
    for i, a in enumerate(angles):
        if a < ANGLE_MIN[i] - 0.5 or a > ANGLE_MAX[i] + 0.5:
            problems.append("%s 超出限位（%.0f 度）" % (JOINT_LABELS[i], a))
    legs = foot_positions(angles, dims)
    ground = min(foot[2] for _, foot in legs)
    # 身体会往低处倾斜，直到再有一只脚着地，所以腿朝下（小于 90 度）的脚都算可能的支撑点
    contacts = [(foot[0], foot[1]) for i, (_, foot) in enumerate(legs)
                if foot[2] - ground < contact_tol or angles[i] < 90]
    # 重心：前半身随腰转，所以扭腰能把重心挪到一边（原厂打招呼动作就靠这个）
    yaw = math.radians(angles[4])
    m = dims["front_mass"]
    com = (-m * dims["front_com_dy"] * math.sin(yaw),
           m * dims["front_com_dy"] * math.cos(yaw) - (1 - m) * dims["rear_com_dy"])
    # 重心和尺寸都是估算的，偏出 tip_margin 以内不报（原厂打招呼动作按估算偏出约 15mm，真机是稳的）
    if _support_margin(com, contacts) < -tip_margin:
        # 四条腿都抬到接近水平时是趴下，身体贴地，不算翻倒
        if not all(a > 70 for a in angles[:4]):
            problems.append("可能翻倒：重心不在着地的脚围成的范围内")
    for left_right in ((0, 2), (1, 3)):
        f, b = legs[left_right[0]], legs[left_right[1]]
        d = _seg_dist_2d((f[0][1], f[0][2]), (f[1][1], f[1][2]), (b[0][1], b[0][2]), (b[1][1], b[1][2]))
        if d < dims["leg_thick"]:
            problems.append("%s 和 %s 可能相撞" % (JOINT_LABELS[left_right[0]], JOINT_LABELS[left_right[1]]))
    return problems


# ---------------------------------------------------------------- 协议

CMD_POSE, CMD_TORQUE, CMD_STREAM = 0x01, 0x02, 0x03
CMD_BUF_RESET, CMD_BUF_APPEND, CMD_BUF_INSTALL, CMD_BUF_PLAY = 0x04, 0x05, 0x06, 0x07
CMD_EXIT, CMD_PING, CMD_PLAY, CMD_LIST = 0x08, 0x09, 0x0A, 0x0B
REP_POSE, REP_ACK, REP_PONG, REP_LIST = 0x81, 0x82, 0x89, 0x8B

BLE_SERVICE = "0000fff0-0000-1000-8000-00805f9b34fb"
BLE_NOTIFY = "0000fff1-0000-1000-8000-00805f9b34fb"
BLE_WRITE = "0000fff2-0000-1000-8000-00805f9b34fb"
BLE_NAME_PREFIX = "RIG-Puppy"


def _le16(v):
    v = int(v) & 0xFFFF
    return bytes([v & 0xFF, v >> 8])


def encode_pose(offsets, ms):
    return bytes([CMD_POSE]) + b"".join(_le16(o) for o in offsets) + _le16(ms)


def ble_frame(payload, order=0x20):
    length = len(payload) + 7
    s = (length + order + sum(payload)) & 0xFF
    return bytes([0x55, 0x00, length, order]) + payload + bytes([0xFF - s, 0x00, 0xAA])


def ble_unframe(data):
    """解析蓝牙通知帧，返回 payload 或 None。"""
    if len(data) < 7 or data[0] != 0x55 or data[1] != 0x00 or data[3] != 0x21:
        return None
    length = data[2]
    if length > len(data):
        return None
    return bytes(data[4:length - 3])


class SerialTransport:
    """USB-C 线（Windows 上是 COMx）。指令行 "@L <hex>"，回复行 "@R <hex>"，其余行是固件日志。"""

    def __init__(self, port, baud=115200):
        import serial  # pyserial
        # 先设好 DTR/RTS 再打开，否则自动复位电路可能让噜噜重启
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = baud
        self.ser.timeout = 0.05
        self.ser.dtr = False
        self.ser.rts = False
        self.ser.open()
        self.on_payload = None
        self.on_log = None
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def send(self, payload):
        with self._lock:
            self.ser.write(b"@L " + payload.hex().encode() + b"\n")

    def _reader(self):
        buf = b""
        while not self._stop:
            try:
                chunk = self.ser.read(256)
            except Exception:
                break
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if line.startswith(b"@R "):
                    try:
                        payload = bytes.fromhex(line[3:].decode())
                    except ValueError:
                        continue
                    if self.on_payload:
                        self.on_payload(payload)
                elif line and self.on_log:
                    self.on_log(line.decode("utf-8", "replace"))

    def close(self):
        self._stop = True
        try:
            self.ser.close()
        except Exception:
            pass


class BleTransport:
    """蓝牙。噜噜要先进入蓝牙遥控模式（对它说“开启蓝牙遥控”）。"""

    def __init__(self, address=None, timeout=8.0):
        import asyncio
        self._asyncio = asyncio
        self.on_payload = None
        self.on_log = None
        self._loop = asyncio.new_event_loop()
        self._queue = queue.Queue()
        self._client = None
        self._ready = threading.Event()
        self._error = None
        self._thread = threading.Thread(target=self._run, args=(address, timeout), daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout + 10):
            raise TimeoutError("蓝牙连接超时")
        if self._error:
            raise self._error

    def _run(self, address, timeout):
        self._asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main(address, timeout))

    async def _main(self, address, timeout):
        from bleak import BleakClient, BleakScanner
        try:
            if address is None:
                dev = await BleakScanner.find_device_by_filter(
                    lambda d, ad: (d.name or "").startswith(BLE_NAME_PREFIX), timeout=timeout)
                if dev is None:
                    raise ConnectionError("没找到噜噜，请确认已开启蓝牙遥控模式")
                address = dev
            self._client = BleakClient(address)
            await self._client.connect()
            await self._client.start_notify(BLE_NOTIFY, self._on_notify)
        except Exception as e:  # noqa: BLE001
            self._error = e
            self._ready.set()
            return
        self._ready.set()
        while self._client.is_connected:
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                await self._asyncio.sleep(0.005)
                continue
            if payload is None:
                break
            try:
                await self._client.write_gatt_char(BLE_WRITE, ble_frame(payload), response=False)
            except Exception as e:  # noqa: BLE001
                if self.on_log:
                    self.on_log("蓝牙发送失败：%s" % e)
        try:
            await self._client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def _on_notify(self, _handle, data):
        payload = ble_unframe(bytes(data))
        if payload and self.on_payload:
            self.on_payload(payload)

    def send(self, payload):
        # 实时姿态只保留最新一条，避免排队造成延迟
        if payload[:1] == bytes([CMD_POSE]):
            try:
                pending = []
                while True:
                    p = self._queue.get_nowait()
                    if p[:1] != bytes([CMD_POSE]):
                        pending.append(p)
            except queue.Empty:
                for p in pending:
                    self._queue.put(p)
        self._queue.put(payload)

    def close(self):
        self._queue.put(None)


class LuluLink:
    """一条到噜噜的连接。所有方法线程安全，可在 Blender 主线程里直接调用。"""

    def __init__(self, transport):
        self.t = transport
        self.t.on_payload = self._on_payload
        self.pose = None          # 最近收到的实际姿态（角度）
        self.pose_time = 0.0
        self.torque_mask = 0x1F
        self.version = None
        self.last_ack = {}
        self.logs = []
        self.t.on_log = self._on_log
        self._ack_event = threading.Event()
        self._closed = False
        self._keepalive = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._keepalive.start()

    @classmethod
    def usb(cls, port, baud=115200):
        return cls(SerialTransport(port, baud))

    @classmethod
    def ble(cls, address=None):
        return cls(BleTransport(address))

    def _on_log(self, line):
        self.logs.append(line)
        del self.logs[:-200]

    def _on_payload(self, p):
        if not p:
            return
        code = p[0]
        if code == REP_POSE and len(p) >= 12:
            offsets = [int.from_bytes(p[1 + i * 2:3 + i * 2], "little", signed=True) for i in range(NUM_JOINTS)]
            self.pose = [offset_to_angle(i, o) for i, o in enumerate(offsets)]
            self.torque_mask = p[11]
            self.pose_time = time.time()
        elif code == REP_ACK and len(p) >= 3:
            self.last_ack[p[1]] = (bool(p[2]), p[3:].decode("utf-8", "replace"))
            self._ack_event.set()
        elif code == REP_PONG:
            self.version = p[1:].decode("utf-8", "replace")
            self._ack_event.set()
        elif code == REP_LIST:
            self.last_ack[CMD_LIST] = (True, p[1:].decode("utf-8", "replace"))
            self._ack_event.set()

    def _keepalive_loop(self):
        # 固件 15 秒收不到指令会自动退出实时操控
        while not self._closed:
            time.sleep(5)
            if not self._closed:
                try:
                    self.t.send(bytes([CMD_PING]))
                except Exception:  # noqa: BLE001
                    pass

    def _request(self, payload, cmd, timeout=2.0):
        self.last_ack.pop(cmd, None)
        self._ack_event.clear()
        self.t.send(payload)
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cmd in self.last_ack:
                return self.last_ack.pop(cmd)
            self._ack_event.wait(0.05)
            self._ack_event.clear()
        return (False, "超时没有回应")

    def ping(self, timeout=2.0):
        self.version = None
        self.t.send(bytes([CMD_PING]))
        deadline = time.time() + timeout
        while time.time() < deadline and self.version is None:
            time.sleep(0.02)
        return self.version

    def send_angles(self, angles, ms=40):
        offsets = [angle_to_offset(i, a) for i, a in enumerate(clamp_angles(angles))]
        self.t.send(encode_pose(offsets, ms))

    def set_torque(self, mask):
        return self._request(bytes([CMD_TORQUE, mask & 0x1F]), CMD_TORQUE)

    def stream(self, hz):
        return self._request(bytes([CMD_STREAM, max(0, min(50, int(hz)))]), CMD_STREAM)

    def exit_live(self):
        return self._request(bytes([CMD_EXIT]), CMD_EXIT)

    def _upload(self, action):
        data = dumps_action(action).encode("utf-8")
        ok, msg = self._request(bytes([CMD_BUF_RESET]), CMD_BUF_RESET)
        if not ok:
            return ok, msg
        chunk = 90 if isinstance(self.t, BleTransport) else 160  # 蓝牙单包受 MTU 限制
        for k in range(0, len(data), chunk):
            ok, msg = self._request(bytes([CMD_BUF_APPEND]) + data[k:k + chunk], CMD_BUF_APPEND)
            if not ok:
                return ok, msg
        return True, ""

    def install_action(self, action):
        ok, msg = self._upload(action)
        return self._request(bytes([CMD_BUF_INSTALL]), CMD_BUF_INSTALL) if ok else (ok, msg)

    def preview_action(self, action, intensity=100, speed=100):
        ok, msg = self._upload(action)
        if not ok:
            return ok, msg
        return self._request(bytes([CMD_BUF_PLAY]) + _le16(intensity) + _le16(speed), CMD_BUF_PLAY)

    def play(self, name):
        return self._request(bytes([CMD_PLAY]) + name.encode("utf-8"), CMD_PLAY)

    def list_actions(self):
        ok, text = self._request(bytes([CMD_LIST]), CMD_LIST)
        return json.loads(text) if ok else []

    def close(self):
        self._closed = True
        try:
            self.t.send(bytes([CMD_EXIT]))
            time.sleep(0.05)
        except Exception:  # noqa: BLE001
            pass
        self.t.close()


def list_serial_ports():
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    return [(p.device, p.description) for p in list_ports.comports()]


# ---------------------------------------------------------------- 命令行

def _main(argv):
    if len(argv) < 2 or argv[1] not in ("ports", "ping", "play", "install"):
        print(__doc__)
        return 1
    if argv[1] == "ports":
        for dev, desc in list_serial_ports():
            print(dev, desc)
        return 0
    port = argv[-1]
    link = LuluLink.usb(port) if not port.lower().startswith("ble") else LuluLink.ble()
    try:
        if argv[1] == "ping":
            print("版本：", link.ping())
        else:
            with open(argv[2], encoding="utf-8") as f:
                action = json.load(f)
            problems = validate_action(action)
            if problems:
                print("\n".join(problems))
                return 1
            print(link.install_action(action) if argv[1] == "install" else link.preview_action(action))
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
