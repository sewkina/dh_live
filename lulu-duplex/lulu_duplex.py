"""噜噜准全双工：设备端回声消除固件 + 服务器打断判定。

固件开了设备端 AEC 时，hello 里会带 features.device_aec=true，并且噜噜说话时麦克风不关
（realtime 聆听）。上游只在“服务器 AEC”时才打断，这里给 device_aec 的连接加上打断：
- 复用 lulu_bargein.BargeInDetector：第一段语音发出后 grace_ms(默认800) 内不打断，连续人声 ≥ min_voice_ms(默认400) 才打断；
- 打断时先发 {"type":"tts","state":"abort"}，固件收到后立刻清掉缓存的语音，
  再走上游 handleAbortMessage（停 LLM/TTS、发 tts stop）。
日志只记“打断”事件，不记音频或文字。
配置（data/.config.yaml，可选）：
  lulu_bargein: {min_voice_ms: 400, grace_ms: 800, max_gap_ms: 120}
关掉：环境变量 LULU_DISABLE_DUPLEX=1
"""

import json
import os
import time

TAG = "lulu_duplex"


def _device_aec(conn) -> bool:
    feats = getattr(conn, "features", None) or {}
    return bool(feats.get("device_aec"))


def install():
    if os.environ.get("LULU_DISABLE_DUPLEX") == "1":
        return
    from core.handle import receiveAudioHandle as rah
    from core.handle import abortHandle
    from core.providers.vad import silero
    from core.utils.lulu_bargein import BargeInDetector

    if getattr(rah.handleAudioMessage, "_lulu_duplex", False):
        return

    # 1) 记下每帧 VAD 结果（不重复跑 VAD，避免打乱 silero 的内部状态）
    vad_cls = silero.VADProvider
    orig_is_vad = vad_cls.is_vad

    def is_vad(self, conn, pcm_frame):
        r = orig_is_vad(self, conn, pcm_frame)
        conn._lulu_last_vad = bool(r)
        return r

    vad_cls.is_vad = is_vad

    # 2) 打断时先让固件清空缓存语音
    orig_abort = abortHandle.handleAbortMessage

    async def abort_with_flush(conn):
        conn._lulu_play_end = 0.0
        conn._lulu_audio_t = None
        if _device_aec(conn):
            try:
                await conn.websocket.send(json.dumps(
                    {"type": "tts", "state": "abort", "session_id": conn.session_id}))
            except Exception:
                pass
        return await orig_abort(conn)

    abortHandle.handleAbortMessage = abort_with_flush
    rah.handleAbortMessage = abort_with_flush

    # 3) 记下“第一段语音真正发给噜噜”的时间。上游在识别出孩子的话时就把
    #    client_is_speaking 设成 True（那时还在等大模型和合成），不能从那时起算打断。
    from core.handle import sendAudioHandle as sah
    orig_send = getattr(sah, "_do_send_audio", None)
    if orig_send is not None:
        async def do_send_audio(conn, *a, **kw):
            now = time.time()
            if getattr(conn, "_lulu_audio_t", None) is None:
                conn._lulu_audio_t = now
            # 估计噜噜实际播放到什么时候（服务器发得比播放快，每包约 60ms）
            end = max(getattr(conn, "_lulu_play_end", 0.0) or 0.0, now)
            conn._lulu_play_end = end + 0.06
            return await orig_send(conn, *a, **kw)
        sah._do_send_audio = do_send_audio

    # 4) device_aec 连接：噜噜真的在出声时，检测到持续人声就打断
    orig_handle = rah.handleAudioMessage

    async def handle_audio(conn, pcm_frame):
        if not _device_aec(conn):
            return await orig_handle(conn, pcm_frame)
        conn._lulu_last_vad = False
        speaking = bool(getattr(conn, "client_is_speaking", False))
        if orig_send is not None:
            # 服务器发完语音就把 client_is_speaking 清掉，但噜噜还在播；按估计的播放结束时间算
            playing = time.time() < (getattr(conn, "_lulu_play_end", 0.0) or 0.0) + 0.3
            if not speaking and not playing:
                conn._lulu_audio_t = None
            speaking = getattr(conn, "_lulu_audio_t", None) is not None and (speaking or playing)
        await orig_handle(conn, pcm_frame)
        det = getattr(conn, "_lulu_bargein", None)
        if det is None:
            cfg = dict((getattr(conn, "config", {}) or {}).get("lulu_bargein", {}) or {})
            cfg.setdefault("grace_ms", 800)
            cfg.setdefault("min_voice_ms", 400)
            det = BargeInDetector.from_config({"lulu_bargein": cfg})
            conn._lulu_bargein = det
        if det.update(speaking and getattr(conn, "client_listen_mode", "") != "manual",
                      conn._lulu_last_vad):
            conn.logger.bind(tag=TAG).info("打断：孩子在噜噜说话时插话")
            await abort_with_flush(conn)

    handle_audio._lulu_duplex = True
    rah.handleAudioMessage = handle_audio
    # asr/base.py 是 from ... import handleAudioMessage，名字要一起换
    from core.providers.asr import base as asr_base
    asr_base.handleAudioMessage = handle_audio
