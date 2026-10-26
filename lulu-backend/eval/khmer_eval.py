"""高棉语 ASR / TTS 摸底评测。

ASR：在 Google FLEURS 高棉语测试集（成人朗读）上算字错率 CER。高棉语词之间不写空格，
所以用 CER，不用 WER；计算前去掉空格和标点。
TTS：用候选 TTS 合成 FLEURS 的句子，再用最好的 ASR 识别回来算 CER（“回读可懂度”），
同时记录实时率 RTF（合成耗时 / 音频时长，<1 才能实时）。

注意：FLEURS 是成人朗读，儿童说话的效果会明显更差，这里只能当上限参考。

用法：
  python eval/khmer_eval.py asr  --parquet fleurs_km_test.parquet -n 30 --models whisper-turbo mms whisper-small-km wav2vec2-km
  python eval/khmer_eval.py asr  --parquet ... --models elevenlabs      # 需要 ELEVENLABS_API_KEY
  python eval/khmer_eval.py tts  --parquet ... -n 15 --tts mms-tts --judge mms
结果追加写到 eval/results.jsonl。
"""

import argparse
import io
import json
import os
import re
import time
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000

ASR_MODELS = {
    "whisper-turbo": ("whisper", "openai/whisper-large-v3-turbo"),
    "whisper-small-km": ("whisper", "seanghay/whisper-small-khmer-v2"),
    "mms": ("mms", "facebook/mms-1b-all"),
    "wav2vec2-km": ("ctc", "gagan3012/wav2vec2-xlsr-khmer"),
    "elevenlabs": ("api", "scribe_v1"),
}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    # 去掉空格、零宽空格、标点（含高棉语句号 ។ 和 ៕）
    text = re.sub(r"[\s​‌‍]", "", text)
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P") and ch not in "។៕៖")


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


def load_fleurs(parquet, n):
    rows = pq.read_table(parquet, columns=["audio", "transcription", "raw_transcription"]).slice(0, n).to_pylist()
    out = []
    for row in rows:
        a, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SR:
            a = resample_poly(a, SR, sr).astype(np.float32)
        out.append((a, row["raw_transcription"] or row["transcription"]))
    return out


class Asr:
    def __init__(self, name):
        import torch
        from transformers import pipeline

        kind, repo = ASR_MODELS[name]
        self.kind, self.name = kind, name
        torch.set_num_threads(os.cpu_count() or 4)
        if kind == "whisper":
            self.pipe = pipeline("automatic-speech-recognition", model=repo, device="cpu")
            self.kw = {"generate_kwargs": {"language": "khmer", "task": "transcribe"}}
        elif kind == "mms":
            from transformers import AutoProcessor, Wav2Vec2ForCTC

            self.proc = AutoProcessor.from_pretrained(repo, target_lang="khm")
            self.model = Wav2Vec2ForCTC.from_pretrained(repo, target_lang="khm", ignore_mismatched_sizes=True)
            self.model.eval()
        elif kind == "ctc":
            self.pipe = pipeline("automatic-speech-recognition", model=repo, device="cpu")
            self.kw = {}
        elif kind == "api":
            self.key = os.environ["ELEVENLABS_API_KEY"]

    def __call__(self, audio):
        if self.kind == "mms":
            import torch

            inputs = self.proc(audio, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                ids = torch.argmax(self.model(**inputs).logits, dim=-1)[0]
            return self.proc.decode(ids)
        if self.kind == "api":
            import requests

            buf = io.BytesIO()
            sf.write(buf, audio, SR, format="WAV")
            r = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": self.key},
                data={"model_id": "scribe_v1", "language_code": "khm"},
                files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                timeout=60,
            )
            r.raise_for_status()
            return r.json().get("text", "")
        try:
            return self.pipe({"raw": audio, "sampling_rate": SR}, **self.kw)["text"]
        except ValueError:
            # 部分社区微调的 whisper 生成配置太旧，不认 language 参数（它们本来就只输出高棉语）
            self.kw = {}
            return self.pipe({"raw": audio, "sampling_rate": SR})["text"]


def synth_mms(text):
    import torch
    from transformers import AutoTokenizer, VitsModel

    if not hasattr(synth_mms, "m"):
        synth_mms.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-khm")
        synth_mms.m = VitsModel.from_pretrained("facebook/mms-tts-khm").eval()
    with torch.no_grad():
        wav = synth_mms.m(**synth_mms.tok(text, return_tensors="pt")).waveform[0].numpy()
    sr = synth_mms.m.config.sampling_rate
    return resample_poly(wav, SR, sr).astype(np.float32) if sr != SR else wav


TTS = {"mms-tts": synth_mms}


def log(rec):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps(rec, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["asr", "tts"])
    ap.add_argument("--parquet", required=True)
    ap.add_argument("-n", type=int, default=30)
    ap.add_argument("--models", nargs="+", default=["whisper-turbo", "mms"])
    ap.add_argument("--tts", nargs="+", default=["mms-tts"])
    ap.add_argument("--judge", default="mms")
    ap.add_argument("--show", type=int, default=2, help="打印前几条识别结果")
    a = ap.parse_args()
    data = load_fleurs(a.parquet, a.n)

    if a.task == "asr":
        for name in a.models:
            asr = Asr(name)
            scores, t0, audio_s = [], time.time(), 0.0
            for i, (audio, ref) in enumerate(data):
                hyp = asr(audio)
                scores.append(cer(ref, hyp))
                audio_s += len(audio) / SR
                if i < a.show:
                    print(f"[{name}] 参考: {ref}\n[{name}] 识别: {hyp}")
            log({"task": "asr", "model": name, "n": len(data), "cer": round(float(np.mean(scores)), 3),
                 "cer_median": round(float(np.median(scores)), 3), "rtf_cpu": round((time.time() - t0) / audio_s, 2)})
    else:
        judge = Asr(a.judge)
        base = [cer(ref, judge(audio)) for audio, ref in data]
        log({"task": "tts_judge_baseline", "judge": a.judge, "n": len(data), "cer_on_real_speech": round(float(np.mean(base)), 3)})
        for name in a.tts:
            scores, synth_s, audio_s = [], 0.0, 0.0
            for i, (_, ref) in enumerate(data):
                t0 = time.time()
                wav = TTS[name](ref)
                synth_s += time.time() - t0
                audio_s += len(wav) / SR
                scores.append(cer(ref, judge(wav)))
                if i == 0:
                    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sample_{name}.wav")
                    sf.write(out, wav, SR)
            log({"task": "tts", "tts": name, "judge": a.judge, "n": len(data),
                 "roundtrip_cer": round(float(np.mean(scores)), 3), "rtf_cpu": round(synth_s / audio_s, 2)})


if __name__ == "__main__":
    main()
