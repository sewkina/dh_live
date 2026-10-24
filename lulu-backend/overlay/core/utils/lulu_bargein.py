"""打断判定（噜噜专用）。

upstream 的逻辑是：服务器端 AEC 开启时，只要有一帧 VAD 判为有人声就立刻打断，
残留回声、咳嗽、拍桌子都会误打断。这里改成：
- TTS 刚开始播放的一小段时间内不打断（AEC 还在收敛）；
- 连续有人声累计到 min_voice_ms 才打断；
- 中间允许短暂的无声帧（孩子说话断断续续），超过 max_gap_ms 才清零。
"""

import time


class BargeInDetector:
    def __init__(self, min_voice_ms=300, grace_ms=400, max_gap_ms=120, frame_ms=60):
        self.min_voice_ms = min_voice_ms
        self.grace_ms = grace_ms
        self.max_gap_ms = max_gap_ms
        self.frame_ms = frame_ms
        self._speaking_since = None
        self._voice_ms = 0
        self._gap_ms = 0

    @classmethod
    def from_config(cls, config: dict):
        cfg = (config or {}).get("lulu_bargein", {}) or {}
        return cls(
            min_voice_ms=int(cfg.get("min_voice_ms", 300)),
            grace_ms=int(cfg.get("grace_ms", 400)),
            max_gap_ms=int(cfg.get("max_gap_ms", 120)),
        )

    def reset(self):
        self._voice_ms = 0
        self._gap_ms = 0

    def update(self, device_speaking: bool, have_voice: bool, now: float = None) -> bool:
        """每收到一帧麦克风音频调用一次，返回 True 表示应该打断。"""
        now = time.time() if now is None else now
        if not device_speaking:
            self._speaking_since = None
            self.reset()
            return False
        if self._speaking_since is None:
            self._speaking_since = now
            self.reset()
        if (now - self._speaking_since) * 1000 < self.grace_ms:
            return False
        if have_voice:
            self._voice_ms += self.frame_ms
            self._gap_ms = 0
        else:
            self._gap_ms += self.frame_ms
            if self._gap_ms > self.max_gap_ms:
                self.reset()
        if self._voice_ms >= self.min_voice_ms:
            self.reset()
            return True
        return False
