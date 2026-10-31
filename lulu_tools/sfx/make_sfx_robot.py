"""合成噜噜的表情音效（机器人风）：清脆的电子“哔啵”声，类似 BB-8、WALL-E。

每个声音由几个“音符”拼成：正弦/三角波，带滑音、颤音，软起音、自然衰减，再加一点短回声。
音高都取自 C 大调五声音阶，听着不刺耳。输出 48kHz 单声道 OGG Opus，每个 0.3～1 秒。
运行：python make_sfx_robot.py [输出目录]      需要 numpy 和 ffmpeg
"""

import math
import os
import subprocess
import sys
import wave

import numpy as np

SR = 48000

# 音名 → 频率
_NOTES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def hz(name):
    """'C6'、'G5' 之类的音名换成 Hz。"""
    n = 12 * (int(name[-1]) + 1) + _NOTES[name[0]] + (1 if "#" in name else 0)
    return 440.0 * 2 ** ((n - 69) / 12)


def _curve(keys, n):
    t = np.linspace(0, 1, n)
    return np.interp(t, [k[0] for k in keys], [k[1] for k in keys])


def note(dur, pitch, wave_="sine", vib=(0.0, 0.0), warble=(0.0, 0.0), attack=0.008, decay=None, amp=1.0):
    """pitch：频率，或关键帧 [(0~1, Hz), ...] 做滑音。
    vib = (Hz, 深度比例) 慢颤音；warble = (Hz, 深度比例) 快速颤动（机器人“咕噜”感）。
    decay：指数衰减速度，None 表示平稳后在结尾收掉。"""
    n = int(SR * dur)
    t = np.arange(n) / SR
    f = _curve(pitch, n) if isinstance(pitch, list) else np.full(n, float(pitch))
    if vib[1]:
        f = f * (1 + vib[1] * np.sin(2 * math.pi * vib[0] * t))
    if warble[1]:
        f = f * (1 + warble[1] * np.sin(2 * math.pi * warble[0] * t))
    ph = 2 * math.pi * np.cumsum(f) / SR
    if wave_ == "sine":
        x = np.sin(ph) + 0.12 * np.sin(2 * ph)          # 一点二次谐波，比纯正弦更“实”
    elif wave_ == "tri":
        x = 2 / math.pi * np.arcsin(np.sin(ph))
    elif wave_ == "soft_square":                        # 圆润的方波（低音“嘟”）
        x = np.tanh(3 * np.sin(ph)) * 0.6
    else:
        raise ValueError(wave_)
    env = np.minimum(1, t / attack)
    if decay:
        env = env * np.exp(-decay * t)
    rel = min(n, int(0.03 * SR))
    env[-rel:] *= np.linspace(1, 0, rel) ** 2
    return x * env * amp


def bell(dur, freq, decay=5.0, amp=0.5):
    """小铃铛：基频 + 不和谐泛音，快速衰减。"""
    n = int(SR * dur)
    t = np.arange(n) / SR
    x = (np.sin(2 * math.pi * freq * t) + 0.35 * np.sin(2 * math.pi * freq * 2.0 * t) * np.exp(-3 * t)
         + 0.2 * np.sin(2 * math.pi * freq * 3.01 * t) * np.exp(-6 * t))
    return x * np.exp(-decay * t) * np.minimum(1, t / 0.002) * amp


def seq(*parts):
    end = max(s + len(p) / SR for s, p in parts)
    out = np.zeros(int(end * SR) + 1)
    for s, p in parts:
        i = int(s * SR)
        out[i:i + len(p)] += p
    return out


def echo(x, delay=0.085, gain=0.22, times=2):
    out = np.concatenate([x, np.zeros(int(SR * delay * times))])
    for k in range(1, times + 1):
        d = int(SR * delay * k)
        out[d:d + len(x)] += x * gain ** k
    return out


def soften(x, width=5):
    """轻微低通，去掉数字味。"""
    k = np.hanning(width + 2)[1:-1]
    return np.convolve(x, k / k.sum(), "same")


def blip(name_or_hz, dur=0.09, up=1.06, **kw):
    """一声“哔”：起音时从略低滑到目标音高。"""
    f = hz(name_or_hz) if isinstance(name_or_hz, str) else name_or_hz
    return note(dur, [(0, f / up), (0.25, f), (1, f)], decay=kw.pop("decay", 9), **kw)


# ---------------------------------------------------------------- 各表情

def happy():      # 哔哔哔↑ 三连上行
    return seq((0, blip("C6")), (0.09, blip("E6")), (0.18, blip("G6", 0.2, decay=7)))


def laughing():   # 嘀嘟嘀嘟 快速跳动
    ns = ["G6", "D6", "A6", "E6", "C7", "G6"]
    return seq(*[(k * 0.075, blip(n, 0.07, decay=14, amp=1 - k * 0.06)) for k, n in enumerate(ns)])


def sad():        # 嘟呜～↓ 慢慢往下滑
    return note(0.8, [(0, hz("E5")), (0.25, hz("E5")), (1, hz("A4"))], "sine", vib=(5, 0.012),
                attack=0.04, decay=1.5)


def angry():      # 嘟嘟嘟↓ 低音不满
    return seq(*[(k * 0.11, note(0.1, [(0, hz("D4") * (1.05 - k * 0.04)), (1, hz("C4") * (1 - k * 0.04))],
                                  "soft_square", warble=(30, 0.04), decay=8)) for k in range(3)])


def surprised():  # 呜咿↑！ 快速上扬
    return seq((0, note(0.2, [(0, hz("C5")), (1, hz("C7"))], "sine", decay=3)),
               (0.2, blip("E7", 0.12, decay=12)))


def sleepy():     # 呼～噗 慢下滑 + 轻轻一声
    return seq((0, note(0.6, [(0, hz("G5")), (1, hz("C5"))], "sine", vib=(3, 0.02), attack=0.08, decay=2.5,
                        amp=0.8)),
               (0.62, note(0.18, hz("C5") * 0.98, "sine", decay=12, amp=0.4)))


def loving():     # 嘟～嘀 两声连音 + 铃
    return seq((0, note(0.2, [(0, hz("G5")), (1, hz("G5"))], vib=(6, 0.01), attack=0.03, decay=2)),
               (0.17, note(0.32, [(0, hz("G5")), (0.2, hz("C6")), (1, hz("C6"))], vib=(6, 0.015), decay=3)),
               (0.42, bell(0.5, hz("E7"), amp=0.25)))


def sing():       # 嘀嗒嘀嗒 小旋律
    ns = ["C6", "E6", "G6", "E6", "C7"]
    return seq(*[(k * 0.12, note(0.14, hz(n), "tri", vib=(7, 0.008), decay=6)) for k, n in enumerate(ns)])


def shy():        # 嘀…嘀 小小声下行
    return seq((0, blip("A6", 0.07, amp=0.55, decay=14)), (0.12, blip("E6", 0.1, amp=0.45, decay=12)))


def dizzy():      # 呜哇呜哇 晃着往下
    n = int(0.75 * SR)
    t = np.arange(n) / SR
    base = np.interp(t, [0, 0.75], [hz("A5"), hz("A4")])
    f = base * (1 + 0.12 * np.sin(2 * math.pi * 6 * t))
    ph = 2 * math.pi * np.cumsum(f) / SR
    env = np.minimum(1, t / 0.02) * np.exp(-1.5 * t)
    env[-1440:] *= np.linspace(1, 0, 1440)
    return (np.sin(ph) + 0.12 * np.sin(2 * ph)) * env


def curious():    # 嘟嘀？ 低一声再上挑
    return seq((0, blip("G5", 0.1, decay=10)),
               (0.12, note(0.24, [(0, hz("C6")), (0.4, hz("C6")), (1, hz("A6"))], decay=4)))


def celebrate():  # 嘀嘀嘀嘀↑ 号角 + 铃
    ns = ["C6", "E6", "G6", "C7"]
    parts = [(k * 0.07, blip(n, 0.08, decay=10)) for k, n in enumerate(ns)]
    parts += [(0.28, note(0.3, hz("C7"), "tri", vib=(8, 0.01), decay=5)),
              (0.3, bell(0.6, hz("G7"), amp=0.3)), (0.4, bell(0.5, hz("E7"), amp=0.25))]
    return seq(*parts)


def goodnight():  # 嘟—嘟—嘟 轻柔下行 + 铃
    ns = ["G5", "E5", "C5"]
    parts = [(k * 0.2, note(0.26, hz(n), "sine", attack=0.03, decay=4, amp=0.7)) for k, n in enumerate(ns)]
    parts.append((0.0, bell(0.9, hz("G6"), decay=3, amp=0.15)))
    return seq(*parts)


def great():      # 叮 叮 叮↑ 铃声
    return seq((0, bell(0.35, hz("C7"))), (0.11, bell(0.35, hz("E7"))), (0.22, bell(0.55, hz("G7"), decay=4)),
               (0.22, blip("C7", 0.12, decay=10, amp=0.4)))


def cute():       # 啵～哔 上挑 + 高音
    return seq((0, note(0.12, [(0, hz("C5")), (1, hz("G5"))], decay=6)),
               (0.13, blip("E7", 0.11, decay=12, amp=0.7)))


SFX = [("happy", happy), ("laughing", laughing), ("sad", sad), ("angry", angry), ("surprised", surprised),
       ("sleepy", sleepy), ("loving", loving), ("sing", sing), ("shy", shy), ("dizzy", dizzy),
       ("curious", curious), ("celebrate", celebrate), ("goodnight", goodnight), ("great", great), ("cute", cute)]


def save(x, path_ogg):
    """峰值归一化后转 Opus。短音效不用 loudnorm（它对 1 秒以内的声音会忽大忽小）。"""
    x = soften(echo(x))
    x = x / (np.abs(x).max() + 1e-9) * 0.8
    pcm = (x * 32767).astype("<i2").tobytes()
    tmp = path_ogg[:-4] + ".wav"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-ar", "48000", "-ac", "1",
                    "-c:a", "libopus", "-b:a", "32k", path_ogg], check=True)
    os.remove(tmp)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, fn in SFX:
        x = fn()
        path = os.path.join(out_dir, "sfx_%s.ogg" % name)
        save(x, path)
        print("%-10s %.2f 秒 %5.1f KB" % (name, len(x) / SR, os.path.getsize(path) / 1024))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out")
