"""用 ElevenLabs 的“文字生成音效”给每个表情生成几版萌宠声，在你自己电脑上跑。

  set ELEVENLABS_API_KEY=你的key          （PowerShell：$env:ELEVENLABS_API_KEY="你的key"）
  python elevenlabs_sfx.py out              → out/happy_1.mp3 … 每个表情 3 版，外加 sfx_happy_1.ogg（噜噜格式）
  python elevenlabs_sfx.py out happy sad    → 只生成这几个
  python elevenlabs_sfx.py out --variants 5 --style robot

挑好以后把喜欢的那版改名成 sfx_<表情名>.ogg，放进固件 main/boards/puppy/ 覆盖，重新编译。
费用：每生成 1 秒约 40 积分（以 ElevenLabs 当前价格为准），全部 15 个 × 3 版大约 1500～2000 积分。
只用 Python 标准库；转 ogg 需要 ffmpeg。
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

API = "https://api.elevenlabs.io/v1/sound-generation"

# 每个表情的描述和时长（秒，最短 0.5）。描述用英文，生成效果更稳定
COMMON = {
    "pet": "tiny cute cartoon puppy, high-pitched, adorable, clean studio recording, no music, no background noise",
    "robot": "tiny cute robot pet like BB-8, soft electronic chirps and beeps, adorable, no music, no background noise",
}
PROMPTS = {
    "happy": ("happy excited short yip, two quick cheerful chirps rising in pitch", 0.6),
    "laughing": ("giggling, a quick series of playful little laughing yips", 0.8),
    "sad": ("sad soft whimper, a gentle descending whine", 1.0),
    "angry": ("grumpy little growl, short and cute, not scary", 0.6),
    "surprised": ("surprised gasp, a quick rising 'woop!'", 0.5),
    "sleepy": ("sleepy little yawn ending in a soft sigh", 1.2),
    "loving": ("affectionate soft coo, warm and sweet", 0.8),
    "sing": ("happy little 'la la la' humming tune, three cheerful notes", 1.0),
    "shy": ("shy tiny squeak, very quiet and soft", 0.5),
    "dizzy": ("dizzy wobbly 'woah woah' sound, swirling pitch", 0.8),
    "curious": ("curious questioning 'hmm?' with rising pitch at the end", 0.6),
    "celebrate": ("joyful cheer 'yay!' with sparkly chime", 1.0),
    "goodnight": ("soft sleepy 'night night' coo with a gentle bell", 1.2),
    "great": ("proud happy bark with a bright ding", 0.8),
    "cute": ("cute 'boop' sound followed by a tiny happy squeak", 0.5),
}


def generate(key, text, seconds, influence):
    body = json.dumps({"text": text, "duration_seconds": max(0.5, seconds), "prompt_influence": influence}).encode()
    req = urllib.request.Request(API, data=body, method="POST",
                                 headers={"xi-api-key": key, "Content-Type": "application/json",
                                          "Accept": "audio/mpeg"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def to_lulu_ogg(src, dst, max_len=1.0):
    """去掉头尾静音、截到 1 秒内、转成噜噜用的 48kHz 单声道 Opus。"""
    af = ("silenceremove=start_periods=1:start_threshold=-45dB,"
          "areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
          "atrim=0:%.2f,afade=t=out:st=%.2f:d=0.06,dynaudnorm=f=75:g=3" % (max_len, max(0, max_len - 0.06)))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, "-ar", "48000", "-ac", "1",
                    "-c:a", "libopus", "-b:a", "32k", dst], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="输出目录")
    ap.add_argument("names", nargs="*", help="只生成这些表情，默认全部")
    ap.add_argument("--variants", type=int, default=3, help="每个表情几版")
    ap.add_argument("--style", default="pet", choices=COMMON, help="pet 小狗叫声 / robot 机器人电子音")
    ap.add_argument("--influence", type=float, default=0.5, help="0～1，越大越贴近描述、变化越少")
    a = ap.parse_args()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("先设置环境变量 ELEVENLABS_API_KEY")
    os.makedirs(a.out, exist_ok=True)
    names = a.names or list(PROMPTS)
    for name in names:
        desc, sec = PROMPTS[name]
        text = "%s, %s" % (desc, COMMON[a.style])
        for k in range(1, a.variants + 1):
            mp3 = os.path.join(a.out, "%s_%d.mp3" % (name, k))
            try:
                data = generate(key, text, sec, a.influence)
            except urllib.error.HTTPError as e:
                sys.exit("%s 生成失败：HTTP %d %s" % (name, e.code, e.read()[:300].decode("utf-8", "replace")))
            with open(mp3, "wb") as f:
                f.write(data)
            to_lulu_ogg(mp3, os.path.join(a.out, "sfx_%s_%d.ogg" % (name, k)))
            print("%s 第 %d 版 → %s" % (name, k, mp3))


if __name__ == "__main__":
    main()
