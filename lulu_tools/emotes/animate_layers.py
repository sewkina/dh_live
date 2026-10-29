"""把分层素材（GPT/Codex 画的 SVG 或 PNG）按 anim.json 做成循环动画，导出 .eaf 和预览 .gif。

用法：python animate_layers.py <素材根目录> [输出目录]
素材根目录下每个子目录是一个表情，里面有 anim.json 和图层文件。格式见同目录的 素材规范.md。

坐标统一用 240×240 的屏幕坐标；所有周期运动的 freq 都是整数，保证首尾无缝循环。
"""

import json
import math
import os
import random
import sys

from PIL import Image, ImageDraw

from make_emotes import SIZE, decode_eaf, write_eaf

SS = 4  # 超采样


def _load(path, size_px):
    """读 SVG 或 PNG，返回 RGBA。full-canvas 图层 size_px = SIZE*SS。"""
    if path.lower().endswith(".svg"):
        import cairosvg  # 只有 SVG 素材才需要
        import io
        png = cairosvg.svg2png(url=path, output_width=size_px, output_height=size_px)
        im = Image.open(io.BytesIO(png)).convert("RGBA")
    else:
        im = Image.open(path).convert("RGBA")
        if im.size != (size_px, size_px):
            im = im.resize((size_px, size_px), Image.LANCZOS)
    return im


def _load_sprite(path, width):
    """贴纸素材：按宽度 width（屏幕像素）缩放，保持比例，裁掉透明边。"""
    if path.lower().endswith(".svg"):
        im = _load(path, 512)
    else:
        im = Image.open(path).convert("RGBA")
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    w = max(1, int(width * SS))
    h = max(1, int(im.height * w / im.width))
    return im.resize((w, h), Image.LANCZOS)


def _wave(t, spec, key="freq"):
    return math.sin(2 * math.pi * (t * spec.get(key, 1) + spec.get("phase", 0)))


def _paste(canvas, im, cx, cy, scale=1.0, angle=0.0, opacity=1.0):
    """以 (cx, cy)（屏幕坐标）为中心贴图。"""
    if opacity <= 0.01 or scale <= 0.01:
        return
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
    if angle:
        im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    if opacity < 1.0:
        a = im.getchannel("A").point(lambda v: int(v * opacity))
        im = im.copy()
        im.putalpha(a)
    x = int(cx * SS - im.width / 2)
    y = int(cy * SS - im.height / 2)
    if x >= canvas.width or y >= canvas.height or x + im.width <= 0 or y + im.height <= 0:
        return  # 完全在画面外
    canvas.alpha_composite(im, (max(0, x), max(0, y)), (max(0, -x), max(0, -y)))


class Layer:
    def __init__(self, spec, folder):
        self.spec = spec
        self.folder = folder
        full = SIZE * SS
        self.images = {}
        if "src" in spec:
            self.images["src"] = self._crop(_load(os.path.join(folder, spec["src"]), full))
        if "blink" in spec:
            self.images["blink"] = self._crop(_load(os.path.join(folder, spec["blink"]["src"]), full))
        if "sprite" in spec or "sprites" in spec:
            names = spec.get("sprites") or [spec["sprite"]]
            self.sprites = [_load_sprite(os.path.join(folder, n), spec.get("size", 16)) for n in names]

    @staticmethod
    def _crop(im):
        bbox = im.getbbox() or (0, 0, 1, 1)
        cx = (bbox[0] + bbox[2]) / 2 / SS
        cy = (bbox[1] + bbox[3]) / 2 / SS
        return im.crop(bbox), cx, cy

    def draw(self, canvas, t):
        s = self.spec
        vis = s.get("visible")
        if vis and not (vis[0] <= t < vis[1]):
            return
        if "src" in s:
            img, cx, cy = self.images["src"]
            b = s.get("blink")
            if b and any(abs(((t - a + 0.5) % 1) - 0.5) < b.get("len", 0.08) / 2 for a in b["at"]):
                img, cx, cy = self.images["blink"]
            dx = dy = 0.0
            if "move" in s:
                m = s["move"]
                dx = m.get("x", 0) * _wave(t, m)
                dy = m.get("y", 0) * _wave(t, {**m, "phase": m.get("phase", 0) + m.get("y_phase", 0)})
                if m.get("bounce"):  # 只往上跳
                    dy = -abs(m.get("y", 0) * _wave(t, m))
            scale = 1.0
            if "scale" in s:
                scale = 1 + s["scale"].get("amp", 0.1) * _wave(t, s["scale"])
            angle = 0.0
            if "rotate" in s:
                r = s["rotate"]
                angle = r.get("amp", 0) * _wave(t, r) + r.get("spin", 0) * t
            opacity = 1.0
            if "opacity" in s:
                o = s["opacity"]
                opacity = o["min"] + (o["max"] - o["min"]) * (0.5 + 0.5 * _wave(t, o))
            if "pivot" in s and (scale != 1.0 or angle):
                # 绕指定点缩放/旋转：先算图层中心相对 pivot 的位移
                px, py = s["pivot"]
                vx, vy = (cx - px) * scale, (cy - py) * scale
                a = math.radians(-angle)
                cx = px + vx * math.cos(a) - vy * math.sin(a)
                cy = py + vx * math.sin(a) + vy * math.cos(a)
            _paste(canvas, img, cx + dx, cy + dy, scale, angle, opacity)

        if "particles" in s:  # 沿直线从 from 飘到 to，循环
            p = s["particles"]
            n = p.get("count", 3)
            for k in range(n):
                u = (t * p.get("speed", 1) + k / n) % 1
                if u < p.get("start", 0.05) or u > p.get("end", 0.92):
                    continue
                x = p["from"][0] + (p["to"][0] - p["from"][0]) * u
                y = p["from"][1] + (p["to"][1] - p["from"][1]) * u
                x += p.get("wobble", 0) * math.sin(2 * math.pi * (u * 2 + k * 0.3))
                lo, hi = p.get("scale", [1, 1])
                fade = min(1.0, min(u, 1 - u) * 6) if p.get("fade", True) else 1.0
                _paste(canvas, self.sprites[k % len(self.sprites)], x, y, lo + (hi - lo) * u,
                       p.get("spin", 0) * u, fade)

        if "orbit" in s:  # 绕一个椭圆转
            o = s["orbit"]
            n = o.get("count", 3)
            for k in range(n):
                a = 2 * math.pi * (t + k / n)
                x = o["center"][0] + o["rx"] * math.cos(a)
                y = o["center"][1] + o["ry"] * math.sin(a)
                lo, hi = o.get("depth_scale", [0.8, 1.1])
                sc = lo + (hi - lo) * (0.5 + 0.5 * math.sin(a))
                _paste(canvas, self.sprites[k % len(self.sprites)], x, y, sc, o.get("spin", 0) * t)

        if "rain" in s:  # 从上往下飘落（彩纸、花瓣、雪）
            r = s["rain"]
            rnd = random.Random(r.get("seed", 7))
            for k in range(r.get("count", 20)):
                x0 = rnd.uniform(*r.get("area", [20, 220]))
                ph = rnd.random()
                u = (t + ph) % 1
                y = -12 + (SIZE + 24) * u
                x = x0 + r.get("wobble", 10) * math.sin(2 * math.pi * (u * 2 + ph))
                ang = r.get("spin", 360) * (u + ph) * (1 if k % 2 else -1)
                _paste(canvas, self.sprites[k % len(self.sprites)], x, y, rnd.uniform(0.8, 1.2), ang)

        if "twinkle" in s:  # 原地一闪一闪
            for k, (x, y, sz) in enumerate(s["twinkle"]["points"]):
                v = max(0.0, _wave(t, {"freq": s["twinkle"].get("freq", 1), "phase": k * 0.23}))
                _paste(canvas, self.sprites[k % len(self.sprites)], x, y, (sz / s.get("size", 16)) * (0.5 + 0.6 * v),
                       0, 0.3 + 0.7 * v)


def render(folder):
    with open(os.path.join(folder, "anim.json"), encoding="utf-8") as f:
        spec = json.load(f)
    layers = [Layer(l, folder) for l in spec["layers"]]
    bg = spec.get("bg", "#0C102C")
    bg_img = None
    if bg.lower().endswith((".svg", ".png")):
        bg_img = _load(os.path.join(folder, bg), SIZE * SS)
    n = spec.get("frames", 30)
    frames = []
    for k in range(n):
        t = k / n
        c = Image.new("RGBA", (SIZE * SS, SIZE * SS), (0, 0, 0, 255))
        disc = Image.new("RGBA", c.size, (0, 0, 0, 0))
        if bg_img is not None:
            disc = bg_img.copy()
        else:
            ImageDraw.Draw(disc).ellipse((0, 0, c.width - 1, c.height - 1), fill=bg)
        c.alpha_composite(disc)
        for layer in layers:
            layer.draw(c, t)
        im = c.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
        mask = Image.new("L", (SIZE, SIZE), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, SIZE - 1, SIZE - 1), fill=255)
        out = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
        out.paste(im, (0, 0), mask)
        frames.append(out)
    return spec, frames


def main(src_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(src_root)):
        folder = os.path.join(src_root, name)
        if not os.path.isfile(os.path.join(folder, "anim.json")):
            continue
        spec, frames = render(folder)
        emote = spec.get("name", name)
        eaf = os.path.join(out_dir, emote + ".eaf")
        write_eaf(eaf, frames)
        back = decode_eaf(eaf)
        back[0].save(os.path.join(out_dir, emote + ".gif"), save_all=True, append_images=back[1:],
                     duration=int(1000 / spec.get("fps", 20)), loop=0, disposal=1)
        print("%-12s %3d 帧 %6.1f KB" % (emote, len(frames), os.path.getsize(eaf) / 1024))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out")
