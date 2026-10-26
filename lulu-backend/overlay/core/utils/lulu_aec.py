"""服务器端回声消除（噜噜专用）。

噜噜的音频是 I2S 麦克风 + 功放，没有硬件回采，板载 AEC 做不了。固件在
“服务器端 AEC”模式下会给每个上行麦克风帧打上时间戳：这个时间戳就是该帧录音时
正在播放的那一帧下行 TTS 音频的时间戳。服务器按时间戳找回自己发出去的 TTS PCM
当参考信号，交给 WebRTC AEC3（livekit 的 AudioProcessingModule）去消回声。

livekit 没装时自动退回 upstream 的谱减法实现（由调用方处理）。
"""

import time
from collections import OrderedDict

import numpy as np

try:
    from livekit.rtc import AudioFrame, AudioProcessingModule

    HAS_APM = True
except Exception:  # pragma: no cover - 依赖缺失时退回
    HAS_APM = False

SAMPLE_RATE = 16000
APM_FRAME = SAMPLE_RATE // 100  # AEC3 每次处理 10ms
CACHE_SECONDS = 5.0


class ServerAec:
    def __init__(self, lead_frames: int = 2, noise_suppression: bool = True):
        if not HAS_APM:
            raise RuntimeError("livekit 未安装，无法使用 AEC3")
        self.apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
            auto_gain_control=False,
        )
        # 设备打的时间戳有 ±1 帧误差，AEC3 要求参考信号必须早于回声，
        # 所以参考帧取“被标记那一帧之后第 lead_frames 帧”，让参考固定领先一点，
        # 剩下的延迟交给 AEC3 自己的延迟估计。
        self.lead_frames = lead_frames
        # timestamp -> (pcm int16 bytes @16k, 写入时间)
        self._ref = OrderedDict()

    def add_reference(self, timestamp: int, pcm: bytes):
        """缓存一帧已经发给设备的 TTS 音频（16k 单声道 int16）。"""
        if timestamp <= 0 or not pcm:
            return
        now = time.time()
        self._ref[timestamp] = (pcm, now)
        while self._ref:
            _, (_, t) = next(iter(self._ref.items()))
            if now - t <= CACHE_SECONDS:
                break
            self._ref.popitem(last=False)

    def process(self, timestamp: int, mic_pcm: bytes) -> bytes:
        """消掉一帧麦克风音频里的回声。timestamp 为 0 表示此刻设备没在播放。"""
        mic = np.frombuffer(mic_pcm, dtype=np.int16)
        ref_entry = None
        if timestamp > 0 and timestamp in self._ref:
            keys = list(self._ref.keys())
            idx = keys.index(timestamp) + self.lead_frames
            if idx < len(keys):
                ref_entry = self._ref[keys[idx]]
        if ref_entry is not None:
            ref = np.frombuffer(ref_entry[0], dtype=np.int16)
        else:
            ref = np.zeros(0, dtype=np.int16)
        if len(ref) < len(mic):
            ref = np.pad(ref, (0, len(mic) - len(ref)))
        else:
            ref = ref[: len(mic)]

        out = np.empty_like(mic)
        n = len(mic) - len(mic) % APM_FRAME
        for i in range(0, n, APM_FRAME):
            ref_frame = AudioFrame(ref[i : i + APM_FRAME].tobytes(), SAMPLE_RATE, 1, APM_FRAME)
            mic_frame = AudioFrame(mic[i : i + APM_FRAME].tobytes(), SAMPLE_RATE, 1, APM_FRAME)
            self.apm.process_reverse_stream(ref_frame)
            self.apm.process_stream(mic_frame)
            out[i : i + APM_FRAME] = np.frombuffer(mic_frame.data, dtype=np.int16)
        out[n:] = mic[n:]
        return out.tobytes()
