"""噜噜准全双工：设备端回声消除固件 + 服务器打断判定。

固件开了设备端 AEC 时，hello 里会带 features.device_aec=true，并且噜噜说话时麦克风不关
（realtime 聆听）。上游只在“服务器 AEC”时才打断，这里给 device_aec 的连接加上打断：
- 复用 lulu_bargein.BargeInDetector：TTS 开头 grace_ms 内不打断，连续人声 ≥ min_voice_ms 才打断；
- 打断时先发 {"type":"tts","state":"abort"}，固件收到后立刻清掉缓存的语音，
  再走上游 handleAbortMessage（停 LLM/TTS、发 tts stop）。
日志只记“打断”事件，不记音频或文字。
配置（data/.config.yaml，可选）：
  lulu_bargein: {min_voice_ms: 300, grace_ms: 400, max_gap_ms: 120}
关掉：环境变量 LULU_DISABLE_DUPLEX=1
"""

import json
import os

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
        if _device_aec(conn):
            try:
                await conn.websocket.send(json.dumps(
                    {"type": "tts", "state": "abort", "session_id": conn.session_id}))
            except Exception:
                pass
        return await orig_abort(conn)

    abortHandle.handleAbortMessage = abort_with_flush
    rah.handleAbortMessage = abort_with_flush

    # 3) device_aec 连接：说话中检测到持续人声就打断
    orig_handle = rah.handleAudioMessage

    async def handle_audio(conn, pcm_frame):
        if not _device_aec(conn):
            return await orig_handle(conn, pcm_frame)
        conn._lulu_last_vad = False
        speaking = bool(getattr(conn, "client_is_speaking", False))
        await orig_handle(conn, pcm_frame)
        det = getattr(conn, "_lulu_bargein", None)
        if det is None:
            det = BargeInDetector.from_config(getattr(conn, "config", {}))
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
