"""画噜噜的新表情动画，导出 .eaf（屏幕用）和 .gif（预览用）。

风格沿用原厂：深蓝圆屏、白色圆环眼睛。每个表情是一段循环动画。
运行：python make_emotes.py [输出目录]
"""

import math
import os
import random
import struct
import sys

from PIL import Image, ImageDraw

SIZE = 240
SS = 3                      # 超采样倍数，画完再缩小，边缘更平滑
BG = (12, 16, 44)           # 原厂背景色
WHITE = (255, 255, 255)
SHADE = (205, 207, 215)     # 眼圈底部阴影
EYE_L, EYE_R, EYE_Y = 70, 170, 112
EYE_R_OUT, EYE_RING = 36, 10
FRAMES = 30
FPS = 20

PINK = (255, 120, 160)
GOLD = (255, 200, 60)
ORANGE = (255, 140, 50)
CYAN = (90, 210, 255)
RED = (240, 60, 80)
GREEN = (110, 220, 140)
PURPLE = (190, 120, 255)


class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (SIZE * SS, SIZE * SS), (0, 0, 0))
        self.d = ImageDraw.Draw(self.im)
        r = SIZE * SS / 2
        self.d.ellipse((0, 0, 2 * r - 1, 2 * r - 1), fill=BG)

    def s(self, v):
        return v * SS

    def circle(self, x, y, r, fill=None, outline=None, width=0):
        x, y, r = self.s(x), self.s(y), self.s(r)
        self.d.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=int(self.s(width)))

    def ellipse(self, x, y, rx, ry, fill):
        x, y, rx, ry = self.s(x), self.s(y), self.s(rx), self.s(ry)
        self.d.ellipse((x - rx, y - ry, x + rx, y + ry), fill=fill)

    def arc(self, x, y, r, start, end, width, fill=WHITE):
        x, y, r = self.s(x), self.s(y), self.s(r)
        self.d.arc((x - r, y - r, x + r, y + r), start, end, fill=fill, width=int(self.s(width)))
        # 圆头端点
        for a in (start, end):
            ax = x + (r - self.s(width) / 2) * math.cos(math.radians(a))
            ay = y + (r - self.s(width) / 2) * math.sin(math.radians(a))
            w = self.s(width) / 2
            self.d.ellipse((ax - w, ay - w, ax + w, ay + w), fill=fill)

    def polygon(self, pts, fill):
        self.d.polygon([(self.s(x), self.s(y)) for x, y in pts], fill=fill)

    def line(self, pts, fill, width):
        self.d.line([(self.s(x), self.s(y)) for x, y in pts], fill=fill, width=int(self.s(width)), joint="curve")

    def finish(self):
        im = self.im.resize((SIZE, SIZE), Image.LANCZOS)
        mask = Image.new("L", (SIZE, SIZE), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, SIZE - 1, SIZE - 1), fill=255)
        out = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
        out.paste(im, (0, 0), mask)
        return out


# ---------------------------------------------------------------- 部件

def ring_eye(c, x, y, r=EYE_R_OUT, px=0, py=0, pupil=8, pupil_color=WHITE, lid=0.0):
    """原厂样式的圆环眼：外圈白环 + 底部阴影 + 中间瞳孔。lid 0~1 为眨眼程度。"""
    c.circle(x, y + 1.5, r, fill=SHADE)
    c.circle(x, y, r, fill=WHITE)
    c.circle(x, y, r - EYE_RING, fill=BG)
    if pupil:
        c.circle(x + px, y + py, pupil, fill=pupil_color)
    if lid > 0:
        top = y - r - 2
        c.polygon([(x - r - 3, top), (x + r + 3, top), (x + r + 3, top + (2 * r + 6) * lid),
                   (x - r - 3, top + (2 * r + 6) * lid)], fill=BG)


def happy_eye(c, x, y, r=30, width=11):
    """开心的 ^ 形眼睛。"""
    c.arc(x, y + 12, r, 200, 340, width)


def closed_eye(c, x, y, r=30, width=10):
    """闭眼（向下弯）。"""
    c.arc(x, y - 14, r, 25, 155, width)


def star(c, x, y, r, fill, rot=0.0, points=5, inner=0.45):
    pts = []
    for k in range(points * 2):
        a = math.radians(rot - 90 + k * 180 / points)
        rr = r if k % 2 == 0 else r * inner
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    c.polygon(pts, fill)


def sparkle(c, x, y, r, fill):
    """四角闪光。"""
    star(c, x, y, r, fill, points=4, inner=0.28)


def heart(c, x, y, r, fill):
    c.circle(x - r * 0.5, y - r * 0.2, r * 0.55, fill=fill)
    c.circle(x + r * 0.5, y - r * 0.2, r * 0.55, fill=fill)
    c.polygon([(x - r * 1.02, y - r * 0.05), (x + r * 1.02, y - r * 0.05), (x, y + r * 0.95)], fill)


def note(c, x, y, s, fill):
    """音符。"""
    c.ellipse(x, y, 6 * s, 4.5 * s, fill)
    c.line([(x + 5 * s, y - 1 * s), (x + 5 * s, y - 22 * s)], fill, 2.6 * s)
    c.polygon([(x + 4 * s, y - 23 * s), (x + 14 * s, y - 17 * s), (x + 14 * s, y - 12 * s),
               (x + 6 * s, y - 16 * s)], fill)


def blush(c, x, y, alpha):
    col = tuple(int(BG[i] + (PINK[i] - BG[i]) * alpha) for i in range(3))
    c.ellipse(x, y, 15, 7, col)


def ease(t):
    return 0.5 - 0.5 * math.cos(2 * math.pi * t)


# ---------------------------------------------------------------- 表情

def sing(t):
    """唱歌：^ ^ 眼随节奏上下，音符从右边飘起来。"""
    c = Canvas()
    bob = 4 * math.sin(2 * math.pi * t * 2)
    happy_eye(c, EYE_L, EYE_Y + bob)
    happy_eye(c, EYE_R, EYE_Y - bob)
    for k, col in enumerate((GOLD, CYAN, PINK)):
        p = (t + k / 3) % 1
        x = 186 + 18 * p + 6 * math.sin(2 * math.pi * (p * 2 + k * 0.3))
        y = 196 - 120 * p
        if 0.05 < p < 0.9:
            note(c, x, y, 0.9 + 0.2 * k, col)
    c.ellipse(120, 168, 9, 7 + 4 * ease(t * 2 % 1), (230, 90, 110))  # 张嘴唱
    return c.finish()


def shy(t):
    """害羞：眼睛往下看，脸颊红晕一闪一闪，偶尔眨眼。"""
    c = Canvas()
    lid = 0.0
    if 0.6 < t < 0.72:
        lid = 1 - abs((t - 0.66) / 0.06)
    look = -3 + 2 * math.sin(2 * math.pi * t)
    ring_eye(c, EYE_L, EYE_Y + 4, r=33, px=look - 4, py=10, pupil=9, lid=max(lid, 0.18))
    ring_eye(c, EYE_R, EYE_Y + 4, r=33, px=look - 4, py=10, pupil=9, lid=max(lid, 0.18))
    a = 0.55 + 0.35 * ease(t)
    blush(c, EYE_L - 6, EYE_Y + 50, a)
    blush(c, EYE_R + 6, EYE_Y + 50, a)
    for dx in (-8, 0, 8):  # 腮红上的小斜线
        c.line([(EYE_L - 6 + dx - 3, EYE_Y + 54), (EYE_L - 6 + dx + 3, EYE_Y + 46)], (255, 180, 200), 2)
        c.line([(EYE_R + 6 + dx - 3, EYE_Y + 54), (EYE_R + 6 + dx + 3, EYE_Y + 46)], (255, 180, 200), 2)
    return c.finish()


def dizzy(t):
    """晕：螺旋眼转圈，小星星绕头转。"""
    c = Canvas()
    for ex in (EYE_L, EYE_R):
        pts = []
        rot = 2 * math.pi * t * (1 if ex == EYE_L else -1)
        for k in range(160):
            a = k / 160 * 2 * math.pi * 3.2
            r = 3 + k / 160 * 31
            sign = 1 if ex == EYE_L else -1
            pts.append((ex + r * math.cos(sign * a + rot), EYE_Y + r * math.sin(sign * a + rot)))
        c.line(pts, WHITE, 6)
    for k in range(3):
        a = 2 * math.pi * (t + k / 3)
        x = 120 + 70 * math.cos(a)
        y = 42 + 14 * math.sin(a)
        star(c, x, y, 9 if math.sin(a) > 0 else 7, GOLD, rot=360 * t)
    c.line([(100, 172), (110, 166), (120, 172), (130, 166), (140, 172)], WHITE, 4)  # 波浪嘴
    return c.finish()


def curious(t):
    """好奇：一只眼睁大，瞳孔左右张望，头顶冒问号。"""
    c = Canvas()
    look = 12 * math.sin(2 * math.pi * t)
    ring_eye(c, EYE_L, EYE_Y + 4, r=30, px=look * 0.8, py=0, pupil=8)
    ring_eye(c, EYE_R, EYE_Y - 2, r=40, px=look, py=-2, pupil=11)
    c.arc(EYE_R, EYE_Y - 52, 26, 220, 320, 6)  # 挑起的眉毛
    s = 1 + 0.1 * math.sin(2 * math.pi * t * 2)
    qx, qy = 194, 58 - 4 * ease(t)
    r = 10 * s
    c.arc(qx, qy - 8 * s, r, 180, 400, 5 * s, GOLD)
    ex = qx + (r - 2.5 * s) * math.cos(math.radians(40))
    ey = qy - 8 * s + (r - 2.5 * s) * math.sin(math.radians(40))
    c.line([(ex, ey), (qx, qy + 4 * s), (qx, qy + 7 * s)], GOLD, 5 * s)
    c.circle(qx, qy + 16 * s, 3.4 * s, fill=GOLD)
    return c.finish()


def celebrate(t):
    """庆祝：^ ^ 眼跳动，彩纸从上面飘下来。"""
    c = Canvas()
    jump = -6 * abs(math.sin(2 * math.pi * t * 2))
    happy_eye(c, EYE_L, EYE_Y + jump, r=32, width=12)
    happy_eye(c, EYE_R, EYE_Y + jump, r=32, width=12)
    c.arc(120, 140 + jump, 22, 20, 160, 7)  # 大笑嘴
    rnd = random.Random(7)
    colors = (GOLD, PINK, CYAN, GREEN, PURPLE, ORANGE)
    for k in range(22):
        x0 = rnd.uniform(20, 220)
        phase = rnd.random()
        p = (t + phase) % 1
        y = -10 + 260 * p
        x = x0 + 10 * math.sin(2 * math.pi * (p * 2 + phase))
        w, h = rnd.uniform(4, 7), rnd.uniform(8, 12)
        a = 2 * math.pi * (p * 3 + phase)
        pts = [(x + dx * math.cos(a) - dy * math.sin(a) * 0.5, y + dx * math.sin(a) + dy * math.cos(a))
               for dx, dy in ((-w, -h / 2), (w, -h / 2), (w, h / 2), (-w, h / 2))]
        c.polygon(pts, colors[k % len(colors)])
    return c.finish()


def goodnight(t):
    """晚安：闭眼慢慢呼吸，月亮和星星一闪一闪。"""
    c = Canvas()
    breathe = 2 * ease(t)
    closed_eye(c, EYE_L, EYE_Y + 6 + breathe, r=28, width=9)
    closed_eye(c, EYE_R, EYE_Y + 6 + breathe, r=28, width=9)
    c.circle(120, 48, 20, fill=GOLD)          # 月亮
    c.circle(130, 41, 18, fill=BG)
    for k, (x, y, r) in enumerate(((60, 50, 6), (180, 40, 7), (200, 80, 5), (40, 90, 4), (160, 70, 4))):
        tw = 0.5 + 0.5 * math.sin(2 * math.pi * (t + k * 0.23))
        col = tuple(int(BG[i] + (255 - BG[i]) * (0.35 + 0.65 * tw)) for i in range(3))
        sparkle(c, x, y, r * (0.7 + 0.5 * tw), col)
    for k in range(2):  # 慢慢飘起的 z
        p = (t + k * 0.5) % 1
        x, y, s = 196 + 10 * p, 112 - 36 * p, 5 + 4 * p
        col = tuple(int(BG[i] + (200 - BG[i]) * (1 - p)) for i in range(3))
        c.line([(x - s, y - s), (x + s, y - s), (x - s, y + s), (x + s, y + s)], col, 2.5)
    return c.finish()


def great(t):
    """真棒：金色星星眼一缩一放，周围闪光。"""
    c = Canvas()
    pulse = 1 + 0.12 * math.sin(2 * math.pi * t * 2)
    for ex in (EYE_L, EYE_R):
        c.circle(ex, EYE_Y + 1.5, EYE_R_OUT, fill=SHADE)
        c.circle(ex, EYE_Y, EYE_R_OUT, fill=WHITE)
        c.circle(ex, EYE_Y, EYE_R_OUT - EYE_RING, fill=BG)
        star(c, ex, EYE_Y + 1, 21 * pulse, GOLD, rot=8 * math.sin(2 * math.pi * t))
        c.circle(ex - 6, EYE_Y - 6, 3, fill=WHITE)
    for k, (x, y) in enumerate(((28, 60), (212, 60), (30, 165), (210, 165), (120, 30))):
        s = max(0.0, math.sin(2 * math.pi * (t * 2 + k * 0.2)))
        if s > 0.05:
            sparkle(c, x, y, 11 * s, (255, 240, 170))
    c.arc(120, 150, 18, 20, 160, 6)
    return c.finish()


def cute(t):
    """卖萌：大黑眼珠带两个高光，轻轻左右歪头，小嘴 ω。"""
    c = Canvas()
    tilt = 5 * math.sin(2 * math.pi * t)
    for k, ex in enumerate((EYE_L, EYE_R)):
        y = EYE_Y + (tilt if k == 0 else -tilt) * 0.6
        c.circle(ex + tilt * 0.4, y + 1.5, 38, fill=SHADE)
        c.circle(ex + tilt * 0.4, y, 38, fill=WHITE)
        c.circle(ex + tilt * 0.4, y + 3, 29, fill=(30, 26, 60))
        c.circle(ex + tilt * 0.4, y + 6, 18, fill=(60, 50, 110))
        sh = 1 + 0.15 * math.sin(2 * math.pi * t * 3)
        c.circle(ex + tilt * 0.4 - 10, y - 8, 8 * sh, fill=WHITE)
        c.circle(ex + tilt * 0.4 + 10, y + 12, 4 * sh, fill=WHITE)
    mx = 120 + tilt * 0.5
    c.arc(mx - 7, 160, 7, 10, 170, 3.5)
    c.arc(mx + 7, 160, 7, 10, 170, 3.5)
    blush(c, 40, 160, 0.55)
    blush(c, 200, 160, 0.55)
    return c.finish()


EMOTES = [
    ("sing", "唱歌", sing),
    ("shy", "害羞", shy),
    ("dizzy", "晕乎乎", dizzy),
    ("curious", "好奇", curious),
    ("celebrate", "庆祝", celebrate),
    ("goodnight", "晚安", goodnight),
    ("great", "真棒", great),
    ("cute", "卖萌", cute),
]


# ---------------------------------------------------------------- EAF 编码

_FRAME_HDR = bytes.fromhex("5a5a5f5300000000000001")   # 魔数 + "_S" + 版本，与原厂文件一致
# 原厂每个 .eaf 末尾都有这段完全相同的信息帧（"_C"，转换工具版本），照抄
_INFO = bytes.fromhex(
    "5a5a5f4300000000000001000000860000006f6e6c696e6520636f6e7665727420302e312e355f323032352d30392d30"
    "3420373c3c456741544d702c4166742374462a28753146212a25573b622f23733731296832426b4d2d74403b49267341"
    "544d517544424d325546295a2f32462975262d312c28493d2f304a4740444b5d543346286f51312b45564f3f43606b75"
    "5a403c2d276d4148")
BLOCK_H = 32


def _palette(img, colors=32):
    q = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = q.getpalette()[:colors * 3]
    pal += [0] * (256 * 3 - len(pal))
    return q, pal


def encode_frame(img):
    q, pal = _palette(img)
    px = q.tobytes()
    blocks = []
    for by in range(0, SIZE, BLOCK_H):
        row = px[by * SIZE:(by + BLOCK_H) * SIZE]
        out = bytearray([0])  # 0 = RLE：(次数, 颜色下标) 成对
        k = 0
        while k < len(row):
            v = row[k]
            n = 1
            while k + n < len(row) and row[k + n] == v and n < 255:
                n += 1
            out += bytes([n, v])
            k += n
        blocks.append(bytes(out))
    hdr = _FRAME_HDR + bytes([8]) + struct.pack("<HHHH", SIZE, SIZE, len(blocks), BLOCK_H)
    hdr += struct.pack("<%dI" % len(blocks), *[len(b) for b in blocks])
    palette = b"".join(bytes([pal[i * 3 + 2], pal[i * 3 + 1], pal[i * 3], 0xFF]) for i in range(256))
    return hdr + palette + b"".join(blocks)


def write_eaf(path, frames):
    datas = [encode_frame(f) for f in frames] + [_INFO]
    table, off = b"", 0
    for d in datas:
        table += struct.pack("<II", len(d), off)
        off += len(d)
    body = table + b"".join(datas)
    header = b"\x89EAF" + struct.pack("<III", len(datas), sum(body) & 0xFFFFFFFF, len(body))
    with open(path, "wb") as f:
        f.write(header + body)


def decode_eaf(path):
    """回读校验用：把 .eaf 解回 RGB 帧。"""
    d = open(path, "rb").read()
    assert d[:4] == b"\x89EAF"
    total, checksum, table_len = struct.unpack_from("<III", d, 4)
    assert checksum == sum(d[16:]) & 0xFFFFFFFF and table_len == len(d) - 16
    base = 16 + total * 8
    frames = []
    for i in range(total):
        size, off = struct.unpack_from("<II", d, 16 + i * 8)
        b = base + off + 2
        if d[b:b + 2] != b"_S":
            continue
        w, h = struct.unpack_from("<HH", d, b + 10)
        nb, _bh = struct.unpack_from("<HH", d, b + 14)
        lens = struct.unpack_from("<%dI" % nb, d, b + 18)
        pal_off = b + 18 + nb * 4
        pal = [(d[pal_off + k * 4 + 2], d[pal_off + k * 4 + 1], d[pal_off + k * 4]) for k in range(256)]
        cur = pal_off + 1024
        px = bytearray()
        for L in lens:
            blk = d[cur + 1:cur + L]
            assert d[cur] == 0
            for k in range(0, len(blk), 2):
                px += bytes([blk[k + 1]]) * blk[k]
            cur += L
        assert len(px) == w * h
        im = Image.new("RGB", (w, h))
        im.putdata([pal[v] for v in px])
        frames.append(im)
    return frames


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, label, fn in EMOTES:
        frames = [fn(k / FRAMES) for k in range(FRAMES)]
        eaf = os.path.join(out_dir, name + ".eaf")
        write_eaf(eaf, frames)
        back = decode_eaf(eaf)
        assert len(back) == FRAMES
        back[0].save(os.path.join(out_dir, name + ".gif"), save_all=True, append_images=back[1:],
                     duration=int(1000 / FPS), loop=0, disposal=1)
        print("%-10s %-4s %6.1f KB" % (name, label, os.path.getsize(eaf) / 1024))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
