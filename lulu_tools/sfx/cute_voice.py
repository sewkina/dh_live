"""把你自己录的声音修成萌宠声，直接生成噜噜能用的音效文件。

  python cute_voice.py 录音.m4a happy               → sfx_happy.ogg
  python cute_voice.py 录音文件夹                     → 文件名就是表情名（happy.m4a → sfx_happy.ogg）
  python cute_voice.py 录音.wav happy --pitch 1.8 --preset baby

处理流程：去掉头尾静音 → 升调（保留速度）→ 稍微加快 → 去低频嗡声 → 加一点颤音 → 截到最长 1 秒 → 统一响度
需要装 ffmpeg（要带 rubberband，官方 Windows 版 full build 自带）。手机录的 m4a、mp3、wav 都行。
"""

import argparse
import os
import subprocess
import sys

PRESETS = {
    # pitch：升调倍数；formant：shifted 连音色一起变（更像小动物），preserved 只变音高（更像小孩）
    "pet": dict(pitch=1.75, tempo=1.15, formant="shifted", vibrato=0.12),
    "baby": dict(pitch=1.4, tempo=1.08, formant="preserved", vibrato=0.06),
    "tiny": dict(pitch=2.2, tempo=1.25, formant="shifted", vibrato=0.15),
}


def process(src, name, out_dir, preset="pet", pitch=None, max_len=1.0):
    p = dict(PRESETS[preset])
    if pitch:
        p["pitch"] = pitch
    out = os.path.join(out_dir, "sfx_%s.ogg" % name)
    trim = ("silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.02,"
            "areverse,silenceremove=start_periods=1:start_threshold=-40dB:start_silence=0.05,areverse")
    af = ",".join([
        trim,
        "highpass=f=180",
        "rubberband=pitch=%.3f:tempo=%.3f:formant=%s" % (p["pitch"], p["tempo"], p["formant"]),
        "vibrato=f=7:d=%.2f" % p["vibrato"],
        "equalizer=f=3000:t=q:w=1:g=3",          # 亮一点
        "atrim=0:%.2f" % max_len,
        "afade=t=out:st=%.2f:d=0.06" % max(0, max_len - 0.06),
        "loudnorm=I=-16:TP=-1.5",
    ])
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, "-ar", "48000", "-ac", "1",
                    "-c:a", "libopus", "-b:a", "24k", out], check=True)
    print("%s → %s" % (src, out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="录音文件或文件夹")
    ap.add_argument("name", nargs="?", help="表情名，例如 happy；src 是文件夹时不用填")
    ap.add_argument("--preset", default="pet", choices=PRESETS)
    ap.add_argument("--pitch", type=float, help="升调倍数，覆盖预设（1.5～2.2 之间试）")
    ap.add_argument("--max", type=float, default=1.0, help="最长秒数")
    ap.add_argument("--out", default=".", help="输出目录")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if os.path.isdir(a.src):
        for f in sorted(os.listdir(a.src)):
            if f.lower().endswith((".wav", ".m4a", ".mp3", ".ogg", ".aac", ".flac")):
                process(os.path.join(a.src, f), os.path.splitext(f)[0], a.out, a.preset, a.pitch, a.max)
    else:
        if not a.name:
            sys.exit("要给出表情名，例如：python cute_voice.py 录音.m4a happy")
        process(a.src, a.name, a.out, a.preset, a.pitch, a.max)


if __name__ == "__main__":
    main()
