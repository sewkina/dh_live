# -*- coding: utf-8 -*-
"""MOSS-Live 实测脚本：把 clips/ 里的 16kHz 单声道 wav 当作“麦克风”实时推给
wss://api.mosi.cn/v1/realtime，记录它听到了什么、回答了什么、多快开口，并把回答语音存下来。

用法（Windows，在本脚本所在文件夹）：
    pip install websockets
    python moss_live_test.py              # 跑全部句子
    python moss_live_test.py --bargein    # 额外测一次“说话中途插话打断”
API Key 只从环境变量 MOSS_API_KEY 或同文件夹的 .env 读取（一行 MOSS_API_KEY=...），不会打印。
"""
import argparse, asyncio, json, os, re, sys, time, wave
from pathlib import Path

import websockets

HERE = Path(__file__).resolve().parent
URL = "wss://api.mosi.cn/v1/realtime?model=moss-live-1.0"
FRAME = 640  # 20ms @16kHz s16le
OUT_RATE = 48000

PERSONA = {
    "name": "噜噜",
    "aliases": ["Lulu", "lulu", "嘀嗲布"],
    "prompt": ("你是噜噜，一只可爱的机器小狗，陪 3 到 8 岁的小朋友聊天、学说话。"
               "说话短、温柔、有趣，每次一两句。"),
}
INSTR = {
    "follow": "对方用什么语言说话，你就用同一种语言回答。对方说高棉语（ភាសាខ្មែរ）时，必须用高棉语回答。",
    "khmer": "无论对方说什么语言，你都只用高棉语（ភាសាខ្មែរ）回答，用高棉文字书写。",
}


def load_key():
    k = os.environ.get("MOSS_API_KEY") or os.environ.get("LULU_MOSS_API_KEY")
    for env in (HERE / ".env", Path(r"D:\Xgo\dh_live\lulu-backend\.env")):
        if k or not env.exists():
            continue
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            s = line.strip()
            if s.startswith("MOSS_API_KEY=") or s.startswith("LULU_MOSS_API_KEY="):
                k = s.split("=", 1)[1].strip().strip('"').strip("'")
    if not k:
        sys.exit("没找到 MOSS_API_KEY：请在本文件夹建 .env，写一行 MOSS_API_KEY=你的key")
    return k


def read_pcm(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, f"{path} 需要 16k 单声道 16bit"
        return w.readframes(w.getnframes())


def script_of(text):
    km = len(re.findall(r"[ក-៿]", text))
    zh = len(re.findall(r"[一-鿿]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    best = max((km, "高棉文"), (zh, "中文"), (lat, "拉丁字母"))
    return best[1] if best[0] else "无"


async def run_clip(key, name, pcm, mode, bargein_pcm=None, wait_after=14.0):
    res = {"clip": name, "mode": mode, "heard": "", "reply": "", "interrupted": "",
           "first_audio_s": None, "first_text_s": None, "status": "", "error": ""}
    audio_out = bytearray()
    t_speech_end = None
    playhead = 0.0          # 模拟播放：本地“播放到”的时间点（monotonic）
    pending_ctx = []
    acks = []               # (到期时间, 消息)
    got_first_audio = asyncio.Event()
    done = asyncio.Event()

    async with websockets.connect(URL, additional_headers={"Authorization": "Bearer " + key},
                                  max_size=None, open_timeout=20) as ws:
        await ws.send(json.dumps({"action": "attach_session"}))
        ready = asyncio.Event()

        async def sender():
            nonlocal t_speech_end
            await ready.wait()
            silence = b"\x00" * FRAME
            t0 = time.monotonic(); n = 0

            async def push(buf):
                nonlocal n
                for i in range(0, len(buf), FRAME):
                    fr = buf[i:i + FRAME].ljust(FRAME, b"\x00")
                    await ws.send(fr)
                    n += 1
                    await asyncio.sleep(max(0, t0 + n * 0.02 - time.monotonic()))
                    flush_acks()

            await push(silence * 50)          # 1 秒静音
            await push(pcm)
            t_speech_end = time.monotonic()
            if bargein_pcm is not None:
                # 等它开口 1.5 秒后插话
                end = time.monotonic() + wait_after
                while not got_first_audio.is_set() and time.monotonic() < end:
                    await push(silence * 5)
                await push(silence * 75)
                res["bargein_at_s"] = round(time.monotonic() - t_speech_end, 2)
                await push(bargein_pcm)
            end = time.monotonic() + wait_after
            while time.monotonic() < end and not done.is_set():
                await push(silence * 5)
            if not done.is_set():
                res["status"] += "timeout "
                await ws.close()

        def flush_acks():
            now = time.monotonic()
            while acks and acks[0][0] <= now:
                _, m = acks.pop(0)
                asyncio.ensure_future(ws.send(json.dumps(m)))

        send_task = asyncio.create_task(sender())
        turns_done = 0
        try:
            async for pkt in ws:
                now = time.monotonic()
                if isinstance(pkt, bytes):
                    if t_speech_end and res["first_audio_s"] is None:
                        res["first_audio_s"] = round(now - t_speech_end, 2)
                        got_first_audio.set()
                    audio_out += pkt
                    dur = len(pkt) / 2 / OUT_RATE
                    playhead = max(playhead, now) + dur
                    if pending_ctx:
                        c = pending_ctx.pop(0)
                        acks.append((playhead, {"action": "playback.chunk.completed", **c}))
                    continue
                msg = json.loads(pkt); a = msg.get("action"); d = msg.get("data") or {}
                if a == "session_attached":
                    await ws.send(json.dumps({"action": "session.update", "session": {
                        "persona": PERSONA, "instructions": INSTR[mode]}}))
                elif a == "session.updated":
                    if d.get("error"):
                        res["error"] = "配置失败: " + d["error"]; break
                    await ws.send(json.dumps({"action": "mic_state", "enabled": True}))
                    ready.set()
                elif a == "input.transcript.done":
                    res["heard"] += (" | " if res["heard"] else "") + d.get("text", "")
                elif a == "output.transcript.updated":
                    if t_speech_end and res["first_text_s"] is None:
                        res["first_text_s"] = round(now - t_speech_end, 2)
                elif a == "output.transcript.done":
                    res["reply"] += (" | " if res["reply"] else "") + d.get("text", "")
                elif a == "response.interrupted":
                    res["interrupted"] = d.get("text", "") or "(是)"
                elif a == "response.audio.context":
                    pending_ctx.append({k: d.get(k) for k in ("chunk_id", "run_id", "turn_id")})
                elif a == "response.audio.done":
                    acks.append((max(playhead, now), {"action": "playback.completed",
                                 "run_id": d.get("run_id"), "turn_id": d.get("turn_id")}))
                    acks.sort(key=lambda x: x[0])
                elif a == "response.audio.stop":
                    acks.clear(); pending_ctx.clear()
                    played = int(max(0, now - (playhead - 0.1)) * 1000)
                    playhead = now
                    await ws.send(json.dumps({"action": "playback.stopped", "run_id": d.get("run_id"),
                                              "turn_id": d.get("turn_id"), "played_audio_ms": played}))
                elif a == "response.ended":
                    res["status"] += (d.get("status") or "") + " "
                    turns_done += 1
                    if bargein_pcm is None or turns_done >= 2:
                        await asyncio.sleep(max(0, playhead - time.monotonic()) + 0.3)
                        flush_acks(); done.set(); break
                elif a == "error":
                    res["error"] = json.dumps(d.get("error") or d, ensure_ascii=False); break
                flush_acks()
        finally:
            send_task.cancel()

    res["reply_script"] = script_of(res["reply"])
    out = HERE / "out" / f"{name}_{mode}{'_bargein' if bargein_pcm else ''}.wav"
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(OUT_RATE); w.writeframes(bytes(audio_out))
    res["audio_file"] = out.name; res["audio_s"] = round(len(audio_out) / 2 / OUT_RATE, 1)
    return res


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="follow,khmer")
    ap.add_argument("--only", default="", help="只跑名字里含这个词的句子，如 km")
    ap.add_argument("--bargein", action="store_true")
    args = ap.parse_args()
    key = load_key()
    clips = sorted((HERE / "clips").glob("*.wav"))
    if args.only:
        clips = [c for c in clips if args.only in c.stem]
    results = []
    for mode in args.modes.split(","):
        for c in clips:
            if mode == "khmer" and not c.stem.startswith("km"):
                continue
            print(f"▶ {c.stem} [{mode}] ...", flush=True)
            try:
                r = await run_clip(key, c.stem, read_pcm(c), mode)
            except Exception as e:
                r = {"clip": c.stem, "mode": mode, "error": f"{type(e).__name__}: {e}"}
            await asyncio.sleep(1)
            print("   听到:", r.get("heard"), "\n   回答:", r.get("reply"),
                  "\n   首音:", r.get("first_audio_s"), "s  文字:", r.get("reply_script"), r.get("error", ""), flush=True)
            results.append(r)
    if args.bargein:
        c1, c2 = HERE / "clips" / "zh_02_story.wav", HERE / "clips" / "zh_01_hello.wav"
        print("▶ 打断测试 ...", flush=True)
        try:
            r = await run_clip(key, "bargein_zh", read_pcm(c1), "follow", bargein_pcm=read_pcm(c2))
        except Exception as e:
            r = {"clip": "bargein_zh", "error": f"{type(e).__name__}: {e}"}
        print("  ", r, flush=True)
        results.append(r)
    stamp = time.strftime("%Y%m%d_%H%M")
    (HERE / "out").mkdir(exist_ok=True)
    p = HERE / "out" / f"results_{stamp}.json"
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("结果已存:", p)


if __name__ == "__main__":
    asyncio.run(main())
