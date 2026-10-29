"""合成噜噜的表情音效：短促的萌宠叫声（啾、呜、哼……），一个表情一个。

声音是“会说元音的小动物”：给定音高曲线和元音（共振峰）曲线，用谐波叠加合成，
再加一点颤音和气声。输出 48kHz 单声道 OGG Opus（和原厂 woof.ogg 一样），每个 0.3～1 秒。
运行：python make_sfx.py [输出目录]      需要 numpy 和 ffmpeg
"""

import math
import os
import subprocess
import sys
import wave

import numpy as np

SR = 48000
# 元音共振峰（Hz），小动物声音整体往上提
VOWELS = {"i": (420, 2700), "u": (380, 1000), "a": (850, 1700), "o": (520, 1050), "e": (550, 2300),
          "m": (300, 1300)}


def curve(keys, n):
    """关键帧折线 → 长度 n 的数组。keys = [(0~1, 值), ...]"""
    t = np.linspace(0, 1, n)
    return np.interp(t, [k[0] for k in keys], [k[1] for k in keys])


def voice(dur, pitch, vowel="i", amp=None, vib=(6.0, 0.02), breath=0.04, bright=1.0, scale=1.25):
    """pitch / amp：关键帧；vowel：元音字母或 [(t, 字母), ...]。"""
    n = int(SR * dur)
    f0 = curve(pitch, n)
    t = np.arange(n) / SR
    f0 = f0 * (1 + vib[1] * np.sin(2 * math.pi * vib[0] * t))
    phase = 2 * math.pi * np.cumsum(f0) / SR
    if isinstance(vowel, str):
        vowel = [(0, vowel), (1, vowel)]
    F1 = curve([(k, VOWELS[v][0] * scale) for k, v in vowel], n)
    F2 = curve([(k, VOWELS[v][1] * scale) for k, v in vowel], n)
    out = np.zeros(n)
    for h in range(1, 24):
        fh = f0 * h
        g = (np.exp(-((fh - F1) / 160) ** 2) + 0.7 * np.exp(-((fh - F2) / 260) ** 2) + 0.06) / h ** (0.6 / bright)
        g[fh > SR / 2 - 2000] = 0
        out += g * np.sin(h * phase)
    noise = np.random.default_rng(1).standard_normal(n)
    noise = np.convolve(noise, np.hanning(15) / np.hanning(15).sum(), "same")   # 去掉刺耳的高频嘶声
    out += breath * 3 * noise * (out.std() + 1e-9)
    env = curve(amp or [(0, 0), (0.08, 1), (0.7, 0.8), (1, 0)], n)
    return out * env


def chime(dur, freqs, decay=6.0, amp=0.5):
    """铃铛/闪光声：几个正弦叠加，快速衰减。"""
    n = int(SR * dur)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, f in enumerate(freqs):
        out += np.sin(2 * math.pi * f * t) * np.exp(-decay * t) / (k + 1) ** 0.5
        out += 0.3 * np.sin(2 * math.pi * f * 2.76 * t) * np.exp(-decay * 2 * t)
    a = np.minimum(1, t / 0.004)
    return out * a * amp


def seq(*parts):
    """[(开始秒, 片段), ...] 混在一起。"""
    end = max(s + len(p) / SR for s, p in parts)
    out = np.zeros(int(end * SR) + 1)
    for s, p in parts:
        i = int(s * SR)
        out[i:i + len(p)] += p
    return out


def happy():      # 啾～啾↑
    return seq((0, voice(0.13, [(0, 900), (1, 1150)], "i")),
               (0.16, voice(0.22, [(0, 1000), (0.6, 1500), (1, 1450)], [(0, "u"), (0.4, "i"), (1, "i")])))


def laughing():   # 嘿嘿嘿嘿
    return seq(*[(k * 0.11, voice(0.09, [(0, 1250 - k * 70), (1, 1150 - k * 70)], "e", breath=0.12)) for k in range(5)])


def sad():        # 呜～↓
    return voice(0.85, [(0, 820), (0.3, 860), (1, 520)], [(0, "u"), (1, "o")], vib=(5.5, 0.035),
                 amp=[(0, 0), (0.15, 0.9), (0.75, 0.6), (1, 0)])


def angry():      # 哼！（低一点，带颤）
    v = voice(0.4, [(0, 360), (0.3, 420), (1, 330)], [(0, "m"), (0.3, "a"), (1, "a")], vib=(28, 0.06),
              breath=0.15, bright=1.6, amp=[(0, 0), (0.05, 1), (0.6, 0.8), (1, 0)])
    return v


def surprised():  # 哇？↑
    return voice(0.35, [(0, 600), (0.7, 1500), (1, 1650)], [(0, "u"), (0.3, "a"), (1, "a")],
                 amp=[(0, 0), (0.1, 1), (0.8, 0.9), (1, 0)])


def sleepy():     # 哈～呼（打哈欠）
    return seq((0, voice(0.6, [(0, 650), (0.35, 900), (1, 450)], [(0, "a"), (0.6, "o"), (1, "u")], breath=0.25,
                         amp=[(0, 0), (0.25, 0.7), (0.7, 0.6), (1, 0)])),
               (0.62, 0.5 * voice(0.3, [(0, 400), (1, 380)], "u", breath=0.6)))


def loving():     # 嗯～嘛 + 闪光
    return seq((0, voice(0.22, [(0, 700), (1, 900)], "m")),
               (0.22, voice(0.25, [(0, 1000), (1, 1250)], [(0, "m"), (0.3, "a"), (1, "a")])),
               (0.45, chime(0.4, [2093, 2637], amp=0.25)))


def sing():       # 啦啦啦（do mi sol）
    notes = [784, 988, 1175, 1568]
    return seq(*[(k * 0.15, voice(0.14, [(0, f), (1, f)], [(0, "m"), (0.25, "a"), (1, "a")], vib=(7, 0.01)))
                 for k, f in enumerate(notes)])


def shy():        # 嘤（小小声往下）
    return 0.7 * voice(0.3, [(0, 1300), (1, 1050)], "i", breath=0.1, amp=[(0, 0), (0.2, 0.8), (1, 0)])


def dizzy():      # 哇呜哇呜（音高晃）
    n = 12
    return voice(0.8, [(k / n, 750 + 250 * (-1) ** k) for k in range(n + 1)],
                 [(k / n, "a" if k % 2 else "u") for k in range(n + 1)], vib=(0, 0))


def curious():    # 嗯？↑
    return voice(0.38, [(0, 650), (0.5, 700), (1, 1250)], [(0, "m"), (0.5, "m"), (1, "e")])


def celebrate():  # 耶！+ 叮叮叮
    return seq((0, voice(0.3, [(0, 900), (0.4, 1400), (1, 1350)], [(0, "e"), (1, "e")])),
               (0.2, chime(0.25, [1568])), (0.3, chime(0.25, [1976])), (0.4, chime(0.5, [2349, 3136])))


def goodnight():  # 晚～安（轻柔下行）+ 铃
    return seq((0, 0.7 * voice(0.3, [(0, 880), (1, 880)], [(0, "u"), (1, "a")], breath=0.1)),
               (0.32, 0.7 * voice(0.45, [(0, 660), (1, 620)], [(0, "a"), (1, "m")], breath=0.1,
                                  amp=[(0, 0), (0.1, 0.8), (1, 0)])),
               (0.1, chime(0.8, [1319], decay=3, amp=0.15)))


def great():      # 叮↑叮↑ + 啾
    return seq((0, chime(0.3, [1319])), (0.12, chime(0.3, [1760])), (0.24, chime(0.5, [2637, 3520])),
               (0.05, 0.8 * voice(0.25, [(0, 1100), (1, 1600)], "i")))


def cute():       # 噗～啾（卖萌）
    return seq((0, voice(0.12, [(0, 700), (1, 650)], "u")),
               (0.13, voice(0.2, [(0, 1400), (0.5, 1700), (1, 1600)], "i")))


SFX = [("happy", happy), ("laughing", laughing), ("sad", sad), ("angry", angry), ("surprised", surprised),
       ("sleepy", sleepy), ("loving", loving), ("sing", sing), ("shy", shy), ("dizzy", dizzy),
       ("curious", curious), ("celebrate", celebrate), ("goodnight", goodnight), ("great", great), ("cute", cute)]


def save(x, path_ogg, path_wav=None):
    x = x / (np.abs(x).max() + 1e-9) * 0.9
    fade = min(len(x), int(0.01 * SR))
    x[-fade:] *= np.linspace(1, 0, fade)
    pcm = (x * 32767).astype("<i2").tobytes()
    tmp = path_wav or path_ogg[:-4] + ".wav"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-af", "loudnorm=I=-16:TP=-1.5",
                    "-ar", "48000", "-ac", "1", "-c:a", "libopus", "-b:a", "24k", path_ogg], check=True)
    if not path_wav:
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
