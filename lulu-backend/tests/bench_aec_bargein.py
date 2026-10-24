"""专项 A 离线测试：模拟噜噜“边说边听”，检查回声消除和打断判定。

用法：python tests/test_aec_bargein.py <fleurs_km_test.parquet> <silero_vad.onnx>

模拟方式：取两段 FLEURS 高棉语录音，一段当噜噜播放的 TTS（参考信号），
一段当孩子插话。麦克风信号 = TTS 经过房间冲激响应 + 延迟 + 喇叭削波后的强回声
+ 噪声 +（2.5 秒后开始的）孩子说话。按固件的方式逐 60ms 帧打时间戳送给服务器。
"""

import io
import os
import sys

import numpy as np
import onnxruntime
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "overlay"))
from core.utils.lulu_aec import ServerAec  # noqa: E402
from core.utils.lulu_bargein import BargeInDetector  # noqa: E402

SR = 16000
FRAME = 960  # 60ms


class Silero:
    def __init__(self, path, threshold=0.5):
        self.s = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros((1, 64), np.float32)
        self.buf = np.zeros(0, np.float32)
        self.th = threshold

    def frame_has_voice(self, pcm):
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32) / 32768])
        voice = False
        while len(self.buf) >= 512:
            chunk, self.buf = self.buf[:512], self.buf[512:]
            x = np.concatenate([self.ctx, chunk[None]], axis=1).astype(np.float32)
            out, self.state = self.s.run(None, {"input": x, "state": self.state, "sr": np.array(SR, np.int64)})
            self.ctx = x[:, -64:]
            voice |= float(out[0][0]) > self.th
        return voice


def load_clips(parquet, n=2, min_sec=6):
    t = pq.read_table(parquet, columns=["audio", "num_samples"]).to_pylist()
    clips = []
    for row in t:
        a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr != SR:
            a = resample_poly(a, SR, sr)
        if len(a) >= min_sec * SR:
            clips.append(a)
        if len(clips) == n:
            break
    return clips


def _norm(x, rms):
    return x / (np.sqrt(np.mean(x**2)) + 1e-9) * rms


def _trim(x, th=0.02):
    # 去掉 FLEURS 录音开头的静音，让“孩子开口时间”可控
    idx = np.argmax(np.abs(x) > th * np.max(np.abs(x)))
    return x[idx:]


def simulate(tts, child, child_start_s=2.5, echo_gain=1.0, child_rms=3000, seed=0):
    rng = np.random.default_rng(seed)
    dur = 6 * SR
    tts = np.pad(_trim(tts), (0, dur))[:dur]
    tts = np.clip(_norm(tts, 4000), -32768, 32767).astype(np.int16)
    # 房间冲激响应：30ms 直达 + 指数衰减混响
    ir = np.zeros(int(0.25 * SR))
    ir[int(0.03 * SR)] = 1.0
    tail = rng.standard_normal(len(ir)) * np.exp(-np.arange(len(ir)) / (0.04 * SR)) * 0.3
    ir += tail
    echo = np.convolve(np.tanh(tts / 32767 * 1.5) * 32767, ir)[:dur] * echo_gain
    near = np.zeros(dur)
    cs = int(child_start_s * SR)
    c = _norm(_trim(child), child_rms)[: dur - cs]
    near[cs : cs + len(c)] = c
    noise = rng.standard_normal(dur) * 60
    mic = np.clip(echo + near + noise, -32768, 32767).astype(np.int16)
    return tts, mic, near.astype(np.int16), echo


def run(parquet, vad_path, pairs=((0, 1), (2, 3), (4, 5), (1, 0))):
    clips = load_clips(parquet, n=6)
    rows = []
    for echo_gain in (2.0, 1.0, 0.5):
        for mode in ("no_aec", "aec3"):
            aborts, onsets, erles = [], [], []
            for a, b in pairs:
                tts16, mic, near, _ = simulate(clips[a], clips[b], echo_gain=echo_gain, seed=a)
                n = len(mic) // FRAME
                e = np.array([np.sqrt(np.mean(near[k * FRAME:(k + 1) * FRAME].astype(float) ** 2)) for k in range(n)])
                onsets.append(round(float(np.argmax(e > 800)) * 0.06, 2))
                aec = ServerAec(lead_frames=1) if mode == "aec3" else None
                if aec:
                    # 服务器发包领先设备播放，参考帧提前全部缓存
                    for j in range(n):
                        aec.add_reference(1000 + j * 60, tts16[j * FRAME:(j + 1) * FRAME].tobytes())
                vad, det, abort_at, out = Silero(vad_path), BargeInDetector(), None, []
                for k in range(n):
                    fr = mic[k * FRAME:(k + 1) * FRAME]
                    # 固件打的是“录音时正在播放那一帧”的时间戳，这里模拟 1 帧误差
                    c = np.frombuffer(aec.process(1000 + max(0, k - 1) * 60, fr.tobytes()), np.int16) if aec else fr
                    out.append(c)
                    if abort_at is None and det.update(True, vad.frame_has_voice(c), now=k * 0.06):
                        abort_at = round(k * 0.06, 2)
                out = np.concatenate(out).astype(np.float64)
                x, y = int(1.0 * SR), int(2.4 * SR)
                erles.append(round(10 * np.log10(np.mean(mic[x:y].astype(np.float64) ** 2) / (np.mean(out[x:y] ** 2) + 1e-9)), 1))
                aborts.append(abort_at)
            rows.append((echo_gain, mode, erles, onsets, aborts))
    return rows


if __name__ == "__main__":
    print("回声强度 | 模式 | 纯回声段抑制(dB) | 孩子开口(s) | 打断时间(s)")
    good = bad = 0
    for eg, mode, erles, onsets, aborts in run(sys.argv[1], sys.argv[2]):
        print(eg, mode, erles, onsets, aborts)
        if mode == "aec3":
            for o, a in zip(onsets, aborts):
                if a is not None and o <= a <= o + 1.0:
                    good += 1
                else:
                    bad += 1
    print(f"AEC3 合格（孩子开口后 1 秒内打断、且没有提前误打断）: {good}/{good + bad}")
