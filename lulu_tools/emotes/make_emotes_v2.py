"""第二批表情：会变形的大眼睛（类似 Vector / EMO 桌面机器人）。

眼睛是参数化的圆角矩形：宽、高、圆角、上眼皮（可斜）、下方笑眼切口、形状（心、星、螺旋……），
每个表情用关键帧驱动这些参数，再叠加嘴、腮红、小装饰。两套配色：
  dark  黑底白眼，眼睛外面一圈随情绪变色的光晕
  white 白底深色眼，带高光
运行：python make_emotes_v2.py [输出目录]      → <输出目录>/dark/*.eaf|gif、<输出目录>/white/*.eaf|gif
"""

import math
import os
import random
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from make_emotes import SIZE, decode_eaf, write_eaf

SS = 4
W = SIZE * SS

THEMES = {
    "dark": dict(bg=(6, 8, 18), ink=(238, 246, 255), hl=None, glow=True, blush=(255, 110, 150),
                 sub=(150, 170, 210)),
    "white": dict(bg=(250, 248, 243), ink=(36, 40, 70), hl=(255, 255, 255), glow=False, blush=(255, 150, 170),
                  sub=(150, 155, 180)),
}
GLOW = dict(calm=(70, 160, 255), warm=(255, 190, 70), love=(255, 90, 150), sad=(80, 120, 255),
            angry=(255, 60, 50), green=(80, 230, 160), purple=(170, 110, 255))
PINK, GOLD, CYAN, RED, GREEN, PURPLE, ORANGE = (255, 110, 160), (255, 200, 50), (80, 210, 255), \
    (245, 60, 70), (100, 225, 140), (180, 120, 255), (255, 140, 50)
CONFETTI = [PINK, GOLD, CYAN, GREEN, PURPLE, ORANGE]

EX, EY = 78, 108      # 左眼中心；右眼 x = 240 - EX
EW, EH = 58, 74


# ---------------------------------------------------------------- 时间曲线

def smooth(u):
    return u * u * (3 - 2 * u)


def kf(t, keys, ease=smooth):
    """关键帧插值。keys = [(时间0~1, 值), ...]，值可以是数或元组。"""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            u = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1
            if isinstance(v0, tuple):
                return tuple(a + (b - a) * u for a, b in zip(v0, v1))
            return v0 + (v1 - v0) * u
    return keys[-1][1]


def blink(t, at, dur=0.08):
    """返回闭眼程度 0~1。at 是一组时间点。"""
    b = 0.0
    for a in at:
        d = abs(((t - a + 0.5) % 1) - 0.5)
        if d < dur / 2:
            b = max(b, 1 - smooth(d / (dur / 2)))
    return b


def wave(t, freq=1, phase=0.0):
    return math.sin(2 * math.pi * (t * freq + phase))


def overshoot(u):
    """弹出来再回一点。"""
    u = max(0.0, min(1.0, u))
    return 1 - math.cos(u * math.pi * 2.5) * (1 - u) ** 2


# ---------------------------------------------------------------- 状态

class Eye:
    def __init__(self, side, **kw):
        self.side = side                      # -1 左，+1 右
        self.x = 120 + side * (120 - EX)
        self.y = EY
        self.w, self.h = EW, EH
        self.r = 0.42                          # 圆角，占短边一半的比例
        self.lid = 0.0                         # 上眼皮盖住的比例
        self.tilt = 0.0                        # 眼皮倾角（度），正值 = 内侧低（生气），负值 = 内侧高（难过）
        self.smile = 0.0                       # 下方切口：0 无，1 很弯的笑眼
        self.shape = "rect"                    # rect / heart / star / spiral / x / arc_down / ring
        self.rot = 0.0                         # 形状旋转（螺旋、星星）
        self.hl = True                         # 白色主题的高光
        self.px = self.py = 0.0                # 瞳孔/高光偏移
        for k, v in kw.items():
            setattr(self, k, v)


class Face:
    def __init__(self, **kw):
        self.L, self.R = Eye(-1), Eye(1)
        self.dx = self.dy = 0.0                # 整张脸平移
        self.tilt = 0.0                        # 歪头（度）
        self.mouth = None                      # (类型, 张开 0~1, 宽度)
        self.mx, self.my = 120, 170
        self.blush = 0.0
        self.glow = GLOW["calm"]
        self.under = []                        # 画在眼睛下面的装饰 fn(draw, theme)
        self.over = []                         # 画在眼睛上面的装饰
        for k, v in kw.items():
            setattr(self, k, v)

    def eyes(self, **kw):
        for e in (self.L, self.R):
            for k, v in kw.items():
                setattr(e, k, v)

    def look(self, gx, gy, persp=0.07):
        """眼神看向 (gx, gy)，-1~1。看哪边哪边的眼睛稍大，像转头。"""
        for e in (self.L, self.R):
            e.x += gx * 16
            e.y += gy * 12
            k = 1 + persp * gx * e.side
            e.w *= k
            e.h *= k
            e.px, e.py = gx * 8, gy * 8

    def blink(self, b):
        if b <= 0:
            return
        for e in (self.L, self.R):
            e.h = e.h * (1 - b) + 7 * b
            e.w *= 1 + 0.12 * b

    def squash(self, s):
        """s > 0 压扁，s < 0 拉长。"""
        for e in (self.L, self.R):
            e.w *= 1 + s
            e.h *= 1 - s


# ---------------------------------------------------------------- 绘制

class Pen:
    """在超采样画布上用屏幕坐标画。"""

    def __init__(self, im):
        self.im = im
        self.d = ImageDraw.Draw(im)

    def circle(self, x, y, r, fill):
        s = SS
        self.d.ellipse(((x - r) * s, (y - r) * s, (x + r) * s, (y + r) * s), fill=fill)

    def ellipse(self, x, y, rx, ry, fill):
        s = SS
        self.d.ellipse(((x - rx) * s, (y - ry) * s, (x + rx) * s, (y + ry) * s), fill=fill)

    def poly(self, pts, fill):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=fill)

    def line(self, pts, fill, width, caps=True):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=int(width * SS), joint="curve")
        if caps:
            for x, y in (pts[0], pts[-1]):
                self.circle(x, y, width / 2, fill)

    def arc(self, x, y, r, a0, a1, width, fill, ry=None):
        ry = r if ry is None else ry
        n = max(8, int(abs(a1 - a0) / 6))
        pts = [(x + r * math.cos(math.radians(a0 + (a1 - a0) * k / n)),
                y + ry * math.sin(math.radians(a0 + (a1 - a0) * k / n))) for k in range(n + 1)]
        self.line(pts, fill, width)

    def rrect(self, x, y, w, h, r, fill):
        s = SS
        rad = max(0, min(w, h) / 2 * r) * s
        self.d.rounded_rectangle(((x - w / 2) * s, (y - h / 2) * s, (x + w / 2) * s, (y + h / 2) * s),
                                 radius=rad, fill=fill)


def star_pts(x, y, r, rot=0.0, points=5, inner=0.48):
    pts = []
    for k in range(points * 2):
        a = math.radians(rot - 90 + k * 180 / points)
        rr = r if k % 2 == 0 else r * inner
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    return pts


def heart(p, x, y, r, fill):
    p.circle(x - r * 0.5, y - r * 0.25, r * 0.56, fill)
    p.circle(x + r * 0.5, y - r * 0.25, r * 0.56, fill)
    p.poly([(x - r * 1.03, y - r * 0.12), (x + r * 1.03, y - r * 0.12), (x, y + r * 0.98)], fill)


def sparkle(p, x, y, r, fill):
    p.poly(star_pts(x, y, r, points=4, inner=0.3), fill)


def note(p, x, y, s, fill):
    p.ellipse(x, y, 6 * s, 4.6 * s, fill)
    p.line([(x + 5 * s, y), (x + 5 * s, y - 22 * s)], fill, 2.8 * s, caps=False)
    p.poly([(x + 4 * s, y - 23 * s), (x + 15 * s, y - 17 * s), (x + 15 * s, y - 11 * s), (x + 6 * s, y - 16 * s)], fill)


def draw_eye(p, e, ink, bg):
    """把一只眼画进单通道蒙版 p（255 = 墨水）。"""
    if e.shape == "rect":
        p.rrect(e.x, e.y, e.w, e.h, max(e.r, 0.5 + 0.5 * e.smile) if e.smile > 0.01 else e.r, 255)
        top = e.y - e.h / 2
        if e.lid > 0.01 or abs(e.tilt) > 0.5:
            # 斜的上眼皮：一条过 (x, top + lid*h) 的直线，内侧（靠鼻子）按 tilt 升降
            yl = top + e.lid * e.h
            k = math.tan(math.radians(e.tilt)) * (-e.side)   # 往内侧（x 朝中心）y 增大 = 生气
            x0, x1 = e.x - e.w, e.x + e.w
            p.poly([(x0, yl + k * (x0 - e.x)), (x1, yl + k * (x1 - e.x)), (x1, top - 60), (x0, top - 60)], 0)
        if e.smile > 0.01:
            # 下方挖掉一个椭圆，变成 ^ 形笑眼
            bot = e.y + e.h / 2
            ry = e.h * 0.85 * e.smile
            p.ellipse(e.x, bot + e.h * 0.08, e.w * 0.6, ry, 0)
    elif e.shape == "heart":
        heart(p, e.x, e.y + 2, e.w * 0.58, 255)
    elif e.shape == "star":
        p.poly(star_pts(e.x, e.y, e.w * 0.6, e.rot), 255)
    elif e.shape == "spiral":
        pts = []
        for k in range(90):
            a = math.radians(e.rot + k * 9 * e.side)
            r = 2 + k * 0.33
            pts.append((e.x + r * math.cos(a), e.y + r * math.sin(a)))
        p.line(pts, 255, 6.5)
    elif e.shape == "x":
        r = e.w * 0.36
        p.line([(e.x - r, e.y - r), (e.x + r, e.y + r)], 255, 11)
        p.line([(e.x - r, e.y + r), (e.x + r, e.y - r)], 255, 11)
    elif e.shape == "arc_down":   # 闭眼 ∪
        p.arc(e.x, e.y - 10, e.w * 0.42, 20, 160, 9, 255, ry=e.w * 0.32)
    elif e.shape == "arc_up":     # 笑眯眯 ∩
        p.arc(e.x, e.y + 14, e.w * 0.44, 200, 340, 10, 255, ry=e.w * 0.4)
    elif e.shape == "chevron":    # > < 用力闭眼
        r = e.w * 0.32
        s = -e.side
        p.line([(e.x - s * r, e.y - r), (e.x + s * r, e.y), (e.x - s * r, e.y + r)], 255, 10)


def draw_mouth(p, f):
    if not f.mouth:
        return
    kind, op, w = f.mouth
    x, y = f.mx + f.dx, f.my + f.dy
    if kind == "smile":
        p.arc(x, y - 8 - 4 * op, w, 25, 155, 6, 255, ry=w * (0.45 + 0.4 * op))
    elif kind == "frown":
        p.arc(x, y + 10, w, 205, 335, 6, 255, ry=w * 0.5)
    elif kind == "flat":
        p.line([(x - w, y), (x + w, y)], 255, 6)
    elif kind == "o":
        p.ellipse(x, y, w * (0.55 + 0.2 * op), w * (0.55 + 0.45 * op), 255)
    elif kind == "open":       # D 形大笑嘴
        h = w * (0.35 + 0.65 * op)
        p.d.chord(((x - w) * SS, (y - h) * SS, (x + w) * SS, (y + h) * SS), 0, 180, fill=255)
    elif kind == "w":          # ω
        r = w / 2
        p.arc(x - r, y - 3, r, 15, 165, 5.5, 255, ry=r * 0.8)
        p.arc(x + r, y - 3, r, 15, 165, 5.5, 255, ry=r * 0.8)
    elif kind == "wave":
        pts = [(x - w + 2 * w * k / 24, y + 4 * math.sin(k / 24 * 4 * math.pi + op * 6.28)) for k in range(25)]
        p.line(pts, 255, 5.5)
    elif kind == "cat":        # :3
        p.arc(x - w / 2, y, w / 2, 0, 180, 5.5, 255, ry=w * 0.35)
        p.arc(x + w / 2, y, w / 2, 0, 180, 5.5, 255, ry=w * 0.35)


def render(face, theme):
    th = THEMES[theme]
    bg, ink = th["bg"], th["ink"]
    # 整张脸平移、歪头：直接改坐标
    for e in (face.L, face.R):
        e.x += face.dx
        e.y += face.dy
        if face.tilt:
            a = math.radians(face.tilt)
            vx, vy = e.x - 120, e.y - 120
            e.x = 120 + vx * math.cos(a) - vy * math.sin(a)
            e.y = 120 + vx * math.sin(a) + vy * math.cos(a)

    mask = Image.new("L", (W, W), 0)
    for e in (face.L, face.R):
        # 每只眼单独画再合并，免得一只眼的眼皮切到另一只
        em = Image.new("L", (W, W), 0)
        draw_eye(Pen(em), e, ink, bg)
        mask = ImageChops.lighter(mask, em)
    draw_mouth(Pen(mask), face)

    under = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    over = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    up, op = Pen(under), Pen(over)
    if face.blush > 0.01:
        col = th["blush"] + (int(200 * min(1, face.blush)),)
        for s in (-1, 1):
            up.ellipse(120 + s * 74 + face.dx, 150 + face.dy, 17, 8, col)
    for fn in face.under:
        fn(up, th)
    for fn in face.over:
        fn(op, th)

    small = mask.resize((SIZE, SIZE), Image.LANCZOS)
    out = Image.new("RGB", (SIZE, SIZE), bg)
    if th["glow"]:
        # 光晕分 3 档，不做连续渐变：屏幕是调色板 + 行程编码，渐变会让文件大好几倍
        halo = small.filter(ImageFilter.GaussianBlur(6)).point(
            lambda v: 0 if v < 24 else 70 if v < 90 else 140 if v < 170 else 210)
        out.paste(Image.new("RGB", (SIZE, SIZE), face.glow), (0, 0), halo)
    u = under.resize((SIZE, SIZE), Image.LANCZOS)
    out.paste(u.convert("RGB"), (0, 0), u.getchannel("A"))
    out.paste(Image.new("RGB", (SIZE, SIZE), ink), (0, 0), small)
    if th["hl"]:
        # 白色主题：大眼睛里点两个高光
        hm = Image.new("L", (W, W), 0)
        hp = Pen(hm)
        for e in (face.L, face.R):
            if e.shape == "rect" and e.hl and e.h > 30 and e.smile < 0.3:
                hx = e.x + e.px * 0.6 - e.w * 0.16
                hy = e.y + e.py * 0.6 - e.h * 0.18 + e.lid * e.h * 0.4
                hp.circle(hx, hy, min(e.w, e.h) * 0.14, 255)
                hp.circle(hx + e.w * 0.24, hy + e.h * 0.24, min(e.w, e.h) * 0.065, 255)
        hm = ImageChops.multiply(hm, mask)
        out.paste(Image.new("RGB", (SIZE, SIZE), th["hl"]), (0, 0), hm.resize((SIZE, SIZE), Image.LANCZOS))
    o = over.resize((SIZE, SIZE), Image.LANCZOS)
    out.paste(o.convert("RGB"), (0, 0), o.getchannel("A"))

    circ = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(circ).ellipse((0, 0, SIZE - 1, SIZE - 1), fill=255)
    final = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    final.paste(out, (0, 0), circ)
    return final


# ---------------------------------------------------------------- 装饰（返回 fn(pen, theme)）

def deco(fn):
    return fn


def rgba(c, a=1.0):
    return tuple(c) + (int(255 * max(0, min(1, a))),)


# ---------------------------------------------------------------- 表情

def neutral(t):
    f = Face()
    gx = kf(t, [(0, 0), (0.15, 0), (0.22, -1), (0.38, -1), (0.45, 0), (0.6, 0), (0.66, 0.8), (0.8, 0.8), (0.86, 0)])
    gy = kf(t, [(0, 0), (0.6, 0), (0.66, -0.3), (0.8, -0.3), (0.86, 0)])
    f.look(gx, gy)
    f.blink(blink(t, [0.5, 0.94, 0.985], 0.06))
    return f


def listen(t):
    f = Face(glow=GLOW["green"])
    f.eyes(w=EW * 1.08, h=EH * 1.08)
    f.tilt = kf(t, [(0, 0), (0.15, -8), (0.85, -8), (1, 0)])
    f.look(0.5, -0.1)
    f.blink(blink(t, [0.62], 0.08))

    def waves(p, th):
        for k in range(3):
            u = (t * 2 + k / 3) % 1
            a = 1 - u
            col = rgba(th["sub"] if not th["glow"] else GLOW["green"], a)
            p.arc(206, 120, 10 + 22 * u, -40, 40, 5, col)
    f.over.append(waves)
    return f


def thinking(t):
    f = Face(glow=GLOW["purple"])
    f.look(0.7, -0.75)
    f.L.lid = 0.3
    f.L.tilt = -8
    f.R.lid = 0.05
    f.mouth = ("flat", 0, 10)
    f.mx = kf(t, [(0, 108), (0.45, 108), (0.55, 128), (0.95, 128), (1, 108)])
    f.blink(blink(t, [0.8], 0.07))

    def dots(p, th):
        n = int(t * 4) % 4
        for k in range(3):
            if k < n:
                col = th["ink"] if not th["glow"] else (200, 180, 255)
                p.circle(176 + k * 16, 52 - k * 6, 5 + k, rgba(col))
    f.over.append(dots)
    return f


def happy(t):
    f = Face(glow=GLOW["warm"])
    jump = abs(wave(t, 2))
    f.dy = -10 * jump
    f.eyes(smile=0.75, h=EH * 0.95)
    f.squash(0.08 * (1 - jump) - 0.04)
    f.mouth = ("smile", 0.3 + 0.3 * jump, 18)
    f.blush = 0.7
    return f


def laughing(t):
    f = Face(glow=GLOW["warm"])
    f.eyes(shape="chevron")
    f.dx = 3 * wave(t, 6)
    f.dy = -3 * abs(wave(t, 6))
    f.mouth = ("open", 0.5 + 0.5 * abs(wave(t, 6)), 26)
    f.my = 168
    f.blush = 0.8

    def tears(p, th):
        for s in (-1, 1):
            u = (t * 2 + (0.5 if s > 0 else 0)) % 1
            x = 120 + s * (60 + 30 * u)
            y = 100 - 20 * math.sin(u * math.pi) + 40 * u
            p.circle(x, y, 5 * (1 - u * 0.5), rgba(CYAN, 1 - u))
    f.over.append(tears)
    return f


def sad(t):
    f = Face(glow=GLOW["sad"])
    f.eyes(lid=0.32, tilt=-18, h=EH * 0.92)
    f.dy = 6 + 2 * wave(t, 1)
    f.look(0, 0.5)
    f.mouth = ("frown", 0, 16)
    f.my = 176 + 2 * wave(t, 4)
    f.blink(blink(t, [0.9], 0.1))

    def tear(p, th):
        u = (t * 1.5) % 1
        if u < 0.85:
            x = f.L.x - 10
            y = f.L.y + f.L.h / 2 + 4 + 60 * smooth(u)
            r = 5 + 2 * min(1, u * 4)
            col = rgba(CYAN, 1 if u < 0.7 else (0.85 - u) / 0.15)
            p.circle(x, y, r, col)
            p.poly([(x - r * 0.9, y - r * 0.3), (x + r * 0.9, y - r * 0.3), (x, y - r * 2.2)], col)
    f.over.append(tear)
    return f


def angry(t):
    f = Face(glow=GLOW["angry"])
    f.eyes(lid=0.32, tilt=24, h=EH * 0.85, w=EW * 1.04)
    shake = wave(t, 8) if 0.3 < t < 0.6 else 0
    f.dx = 4 * shake
    f.squash(0.06 * abs(wave(t, 4)))
    f.mouth = ("frown", 0, 18 + 4 * abs(shake))

    def vein(p, th):
        s = 1 + 0.25 * abs(wave(t, 4))
        x, y = 190, 52
        col = rgba(RED)
        for a in (45, 135, 225, 315):
            dx, dy = math.cos(math.radians(a)), math.sin(math.radians(a))
            p.arc(x + dx * 10 * s, y + dy * 10 * s, 7 * s, a + 135, a + 225, 4.5, col)

    def steam(p, th):
        for k, sx in enumerate((-1, 1)):
            u = (t * 2 + k * 0.5) % 1
            p.circle(120 + sx * (100 - 10 * u), 150 - 40 * u, 4 + 8 * u, rgba(th["sub"], 0.8 * (1 - u)))
    f.over += [vein, steam]
    return f


def surprised(t):
    f = Face(glow=GLOW["warm"])
    pop = overshoot(min(1, t / 0.25))
    s = 0.75 + 0.35 * pop
    f.eyes(w=EW * s * 0.95, h=EH * s, r=0.9)
    f.dy = -6 * pop
    f.mouth = ("o", 0.3 + 0.4 * pop, 12)
    f.my = 172
    f.blink(blink(t, [0.85], 0.06))

    def bang(p, th):
        a = min(1, t / 0.2)
        y = 46 - 6 * pop
        col = rgba(GOLD, a)
        p.poly([(196, y - 22), (208, y - 22), (204, y + 4), (200, y + 4)], col)
        p.circle(202, y + 13, 5, col)
    f.over.append(bang)
    return f


def sleepy(t):
    f = Face(glow=GLOW["sad"])
    droop = kf(t, [(0, 0.35), (0.35, 0.75), (0.5, 0.95), (0.62, 0.4), (0.75, 0.45), (1, 0.35)])
    f.eyes(lid=droop, tilt=-6)
    f.dy = kf(t, [(0, 0), (0.5, 10), (0.62, 0), (1, 0)])
    f.mouth = ("o", 0.2 * blink(t, [0.68], 0.25), 8)
    f.my = 178

    def bubble(p, th):
        u = (t * 1) % 1
        r = 4 + 16 * smooth(min(1, u / 0.5)) if u < 0.5 else 0
        if r:
            col = rgba(CYAN if th["glow"] else (140, 190, 230), 0.8)
            p.circle(150, 168, r, col)
            p.circle(150 - r * 0.3, 168 - r * 0.3, r * 0.25, rgba((255, 255, 255), 0.9))
    f.over.append(bubble)
    return f


def loving(t):
    f = Face(glow=GLOW["love"])
    beat = max(abs(wave(t, 2)), 0) ** 3
    f.eyes(shape="heart", w=EW * (1 + 0.18 * beat))
    f.mouth = ("smile", 0.2, 14)
    f.blush = 0.9

    def hearts(p, th):
        for k in range(4):
            u = (t + k / 4) % 1
            x = 30 + k * 60 + 8 * wave(u, 2)
            y = 230 - 220 * u
            heart(p, x, y, 7 + 3 * k % 3, rgba(PINK, min(1, (1 - u) * 3, u * 6)))
    f.under.append(hearts)
    # 心形眼也上色
    f.eye_color = PINK
    return f


def sing(t):
    f = Face(glow=GLOW["warm"])
    f.eyes(smile=0.7)
    f.tilt = 8 * wave(t, 1)
    f.dy = -4 * abs(wave(t, 2))
    f.mouth = ("o", abs(wave(t, 4)), 13)
    f.my = 172

    def notes(p, th):
        cols = [GOLD, CYAN, PINK]
        for k in range(3):
            u = (t + k / 3) % 1
            a = min(1, u * 6, (1 - u) * 5)
            note(p, 192 + 8 * wave(u, 1.5), 196 - 130 * u, 1.0 + 0.3 * u, rgba(cols[k], a))
    f.over.append(notes)
    return f


def shy(t):
    f = Face(glow=GLOW["love"])
    peek = kf(t, [(0, 0), (0.3, 0), (0.4, 1), (0.6, 1), (0.7, 0), (1, 0)])
    f.look(-0.8 + 1.0 * peek, 0.7 - 0.6 * peek)
    f.eyes(smile=0.7 * (1 - peek), w=EW * 0.9, h=EH * (0.75 + 0.2 * peek), r=0.8)
    f.blink(blink(t, [0.82], 0.08))
    f.blush = 0.6 + 0.4 * abs(wave(t, 2))
    f.mouth = ("wave", t * 2, 10)
    f.my = 174
    f.tilt = -6

    def lines(p, th):
        for s in (-1, 1):
            for k in range(3):
                x = 120 + s * (66 + k * 8)
                p.line([(x + 3, 145), (x - 3, 155)], rgba((255, 255, 255) if th["glow"] else (230, 90, 120), 0.8), 2.5)
    f.over.append(lines)
    return f


def dizzy(t):
    f = Face(glow=GLOW["purple"])
    f.eyes(shape="spiral", rot=-720 * t)
    f.tilt = 10 * wave(t, 1)
    f.dx = 6 * wave(t, 1, 0.25)
    f.mouth = ("wave", t * 2, 16)
    f.my = 176

    def stars(p, th):
        for k in range(3):
            a = 2 * math.pi * (t + k / 3)
            x = 120 + 70 * math.cos(a)
            y = 40 + 12 * math.sin(a)
            p.poly(star_pts(x, y, 8 + 3 * math.sin(a), 360 * t), rgba(GOLD))
    f.over.append(stars)
    return f


def curious(t):
    f = Face(glow=GLOW["green"])
    gx = kf(t, [(0, -0.8), (0.3, -0.8), (0.4, 0.8), (0.7, 0.8), (0.8, -0.8), (1, -0.8)])
    f.look(gx, -0.2)
    f.L.w *= 0.86
    f.L.h *= 0.86
    f.L.lid = 0.12
    f.L.tilt = -10
    f.R.w *= 1.12
    f.R.h *= 1.12
    f.tilt = kf(t, [(0, -7), (0.3, -7), (0.4, 7), (0.7, 7), (0.8, -7), (1, -7)])
    f.blink(blink(t, [0.55], 0.06))
    f.mouth = ("o", 0, 7)
    f.mx = 126

    def q(p, th):
        y = 46 - 6 * blink(t, [0.4, 0.8], 0.1)
        col = rgba(GOLD)
        p.arc(198, y - 6, 9, 180, 405, 5.5, col)
        p.line([(198 + 9 * math.cos(math.radians(45)), y - 6 + 9 * math.sin(math.radians(45))), (198, y + 8)], col, 5.5)
        p.circle(198, y + 18, 3.5, col)
    f.over.append(q)
    return f


def celebrate(t):
    f = Face(glow=GLOW["warm"])
    jump = abs(wave(t, 2))
    f.dy = -12 * jump
    f.eyes(shape="star", w=EW * (0.95 + 0.15 * jump), rot=15 * wave(t, 2))
    f.mouth = ("open", 0.6 + 0.4 * jump, 22)
    f.my = 170
    f.blush = 0.6

    def confetti(p, th):
        rnd = random.Random(5)
        for k in range(22):
            x0 = rnd.uniform(20, 220)
            ph = rnd.random()
            u = (t + ph) % 1
            y = -10 + 260 * u
            x = x0 + 10 * math.sin(2 * math.pi * (u * 2 + ph))
            a = math.radians(360 * (u + ph) * (1 if k % 2 else -1))
            dx, dy = 5 * math.cos(a), 2.5 * math.sin(a)
            col = rgba(CONFETTI[k % len(CONFETTI)])
            p.poly([(x - dx - dy, y - dy + dx), (x + dx - dy, y + dy + dx), (x + dx + dy, y + dy - dx),
                    (x - dx + dy, y - dy - dx)], col)
    f.under.append(confetti)
    f.eye_color = GOLD
    return f


def goodnight(t):
    f = Face(glow=GLOW["sad"])
    f.eyes(shape="arc_down")
    f.dy = 3 * wave(t, 1)
    f.mouth = ("smile", 0, 8)
    f.my = 168

    def sky(p, th):
        mc = (255, 215, 110)
        p.circle(54, 52, 18, rgba(mc))
        p.circle(64, 46, 16, rgba(th["bg"]))
        for k, (x, y, r) in enumerate([(98, 36, 5), (150, 30, 6), (186, 62, 5), (34, 96, 4)]):
            v = max(0, wave(t, 2, k * 0.27))
            sparkle(p, x, y, r * (0.6 + 0.6 * v), rgba((255, 240, 180), 0.35 + 0.65 * v))
        for k in range(3):
            u = (t + k / 3) % 1
            s = 7 + 7 * u
            x, y = 170 + 30 * u, 150 - 70 * u
            col = rgba(th["sub"] if not th["glow"] else (180, 200, 255), min(1, u * 5, (1 - u) * 4))
            p.line([(x - s / 2, y - s / 2), (x + s / 2, y - s / 2), (x - s / 2, y + s / 2), (x + s / 2, y + s / 2)],
                   col, 3, caps=False)
    f.under.append(sky)
    return f


def great(t):
    f = Face(glow=GLOW["warm"])
    pulse = abs(wave(t, 2))
    f.eyes(smile=0.6, w=EW * (1 + 0.06 * pulse), h=EH * (1 + 0.06 * pulse))
    f.dy = -4 * pulse
    f.mouth = ("smile", 0.8, 20)

    def bursts(p, th):
        pts = [(36, 60), (204, 56), (40, 176), (200, 178), (120, 30)]
        for k, (x, y) in enumerate(pts):
            v = max(0, wave(t, 2, k * 0.2))
            sparkle(p, x, y, 4 + 10 * v, rgba(GOLD, 0.3 + 0.7 * v))
        # 大拇指：用一个金色大星星代替，转着出现
        s = overshoot(min(1, t / 0.3))
        p.poly(star_pts(120, 214, 14 * s, 20 * wave(t, 1)), rgba(GOLD))
    f.over.append(bursts)
    return f


def cute(t):
    f = Face(glow=GLOW["love"])
    f.eyes(w=EW * 1.1, h=EH * 1.12, r=0.95)
    f.tilt = kf(t, [(0, -8), (0.45, -8), (0.55, 8), (0.95, 8), (1, -8)])
    wink = blink(t, [0.3], 0.14)
    if wink > 0.5:
        f.R.shape = "arc_up"
    f.blink(blink(t, [0.75], 0.07))
    f.mouth = ("w", 0, 16)
    f.my = 166
    f.blush = 0.85

    def spark(p, th):
        if wink > 0.2:
            sparkle(p, 196, 70, 10 * wink, rgba(GOLD))
    f.over.append(spark)
    return f


# (名字, 中文, 函数, 循环秒数, 帧率)。慢悠悠的表情用 10 帧/秒，文件小一半
EMOTES = [
    ("neutral", "平静", neutral, 3, 10), ("listen", "在听", listen, 2, 10), ("thinking", "思考", thinking, 3, 10),
    ("happy", "开心", happy, 1.5, 20), ("laughing", "大笑", laughing, 1.5, 15), ("sad", "难过", sad, 2.4, 10),
    ("angry", "生气", angry, 2, 12), ("surprised", "惊讶", surprised, 2, 15), ("sleepy", "困", sleepy, 3, 10),
    ("loving", "喜爱", loving, 2, 10),
    ("sing", "唱歌", sing, 2, 10), ("shy", "害羞", shy, 2.5, 10), ("dizzy", "晕乎乎", dizzy, 1.5, 12),
    ("curious", "好奇", curious, 3, 10), ("celebrate", "庆祝", celebrate, 2, 10), ("goodnight", "晚安", goodnight, 3, 10),
    ("great", "真棒", great, 2, 10), ("cute", "卖萌", cute, 2.5, 12),
]


def frame(fn, t, theme):
    f = fn(t)
    th = THEMES[theme]
    col = getattr(f, "eye_color", None)
    if col and theme == "dark":
        # 心形、星星眼直接上色
        saved = th["ink"]
        th["ink"] = col
        im = render(f, theme)
        th["ink"] = saved
        return im
    if col and theme == "white":
        saved = th["ink"]
        th["ink"] = tuple(int(c * 0.95) for c in col)
        im = render(f, theme)
        th["ink"] = saved
        return im
    return render(f, theme)


def main(out_root, only=None):
    for theme in THEMES:
        out_dir = os.path.join(out_root, theme)
        os.makedirs(out_dir, exist_ok=True)
        total = 0
        for name, label, fn, sec, fps in EMOTES:
            if only and name not in only:
                continue
            n = int(round(sec * fps))
            frames = [frame(fn, k / n, theme) for k in range(n)]
            eaf = os.path.join(out_dir, name + ".eaf")
            write_eaf(eaf, frames, dedupe=True)
            back = decode_eaf(eaf)
            assert len(back) == n
            back[0].save(os.path.join(out_dir, name + ".gif"), save_all=True, append_images=back[1:],
                         duration=int(1000 / fps), loop=0, disposal=1)
            kb = os.path.getsize(eaf) / 1024
            total += kb
            print("%-6s %-10s %-4s %3d 帧 %2d fps %6.1f KB" % (theme, name, label, n, fps, kb))
        print("%s 合计 %.1f KB" % (theme, total))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out_v2", sys.argv[2:] or None)
