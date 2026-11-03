# -*- coding: utf-8 -*-
"""MOSS 音色设计 + 语音合成实测：设计一个“噜噜”可爱声音，拿到 voice_id，
再用 moss-tts-1.5-flash 念中文和高棉语，看高棉语能不能念（官方语言表里没有高棉语）。
用法：python moss_voice_test.py      结果存到 out_voice/
Key 读取方式与 moss_live_test.py 相同（.env 里的 LULU_MOSS_API_KEY），不会打印。
"""
import json, sys, time
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_live_test import load_key  # noqa: E402

API = "https://api.mosi.cn/v1"
OUT = Path(__file__).resolve().parent / "out_voice"
INSTR = ("一只可爱的机器小狗的声音，像五六岁的小朋友，奶声奶气、活泼、带笑意，"
         "音调偏高但不尖，说话清楚，语速稍慢")
TEXTS = {
    "zh": "你好呀，我是噜噜！今天想跟我玩什么游戏呢？",
    "km": "សួស្តី! ខ្ញុំឈ្មោះ លូលូ។ ថ្ងៃនេះ យើងលេងអ្វីជាមួយគ្នា?",
    "en": "Hi! I'm Lulu. What game do you want to play today?",
}


def post(path, body, key, tries=3):
    for i in range(tries):
        try:
            return _post(path, body, key)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3)


def _post(path, body, key):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        ct, data = r.headers.get("Content-Type", ""), r.read()
    return ct, data, round(time.time() - t, 2)


def save(name, ct, data, key):
    if "json" in ct:
        j = json.loads(data)
        url = j.get("url") or (j.get("data") or {}).get("url") or j.get("audio_url")
        (OUT / f"{name}.json").write_text(json.dumps(j, ensure_ascii=False, indent=2), encoding="utf-8")
        if url:
            urllib.request.urlretrieve(url, OUT / f"{name}.mp3")
        return j
    (OUT / f"{name}.mp3").write_bytes(data)
    return {}


def main():
    key = load_key(); OUT.mkdir(exist_ok=True); log = []
    voice_id = None
    for lang, text in TEXTS.items():
        try:
            body = {"model": "moss-voice-generator-1.0", "input": text, "instruction": INSTR,
                    "response_format": "mp3", "delivery_method": "url"}
            if lang == "zh":
                body["create_voice"] = True
            ct, data, s = post("/audio/voice/generations", body, key)
            j = save(f"design_{lang}", ct, data, key)
            voice_id = voice_id or j.get("voice_id") or (j.get("data") or {}).get("voice_id")
            log.append(f"音色设计 {lang}: {s}s")
        except Exception as e:
            log.append(f"音色设计 {lang} 失败: {e}")
    if voice_id:
        (OUT / "voice_id.txt").write_text(voice_id)
        for lang, text in TEXTS.items():
            try:
                ct, data, s = post("/audio/speech", {"model": "moss-tts-1.5-flash", "input": text,
                                   "voice_id": voice_id, "response_format": "mp3"}, key)
                save(f"tts_{lang}", ct, data, key); log.append(f"TTS {lang}: {s}s")
            except Exception as e:
                log.append(f"TTS {lang} 失败: {e}")
    else:
        log.append("没拿到 voice_id，看 out_voice/design_zh.json")
    (OUT / "log.txt").write_text("\n".join(log), encoding="utf-8")
    print("\n".join(log)); print("结果在", OUT)


if __name__ == "__main__":
    main()
