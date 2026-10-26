# 噜噜自建后端（lulu-backend）

在 [xiaozhi-esp32-server](https://github.com/xinnan-tech/xiaozhi-esp32-server) 官方镜像上叠加噜噜需要的改动：

- ElevenLabs 高棉语语音识别（Scribe）和语音合成 provider；
- 大模型、视觉模型都走 LiteLLM 网关（`lulu-chat` / `lulu-vision`），换模型不用动固件；
- 可选的 vLLM / SGLang 自建模型（需要 GPU 服务器）；
- 儿童向角色提示词、VAD 句尾静音放长到 1.2 秒；
- 服务器端回声消除（WebRTC AEC3）和防误打断的实验模块（还没有接进主流程，见下）。

## 目录

| 路径 | 作用 |
| --- | --- |
| `docker-compose.yml` | xiaozhi-server + LiteLLM；`--profile gpu` 再加 vLLM |
| `Dockerfile` | 在官方镜像上覆盖 `overlay/` 的文件 |
| `data/config.template.yaml` | 噜噜的配置，`${变量}` 启动时从 `.env` 填入 |
| `gateway/litellm.yaml` | 网关：`lulu-chat`、`lulu-vision` 背后接哪个模型 |
| `scripts/sglang_start.sh` | 备选：用 SGLang 起自建模型 |
| `overlay/core/providers/asr/elevenlabs.py` | ElevenLabs Scribe 识别，默认 `khm` |
| `overlay/core/providers/tts/elevenlabs.py` | ElevenLabs 合成，填噜噜音色的 voice_id |
| `overlay/core/utils/lulu_aec.py`、`lulu_bargein.py` | 回声消除 + 打断判定（实验） |
| `tests/bench_aec_bargein.py` | 回声消除/打断离线模拟 |
| `eval/khmer_eval.py` | 高棉语 ASR / TTS 摸底评测 |

## 部署（服务器：Ubuntu 22.04，4 核 8G，开放 8000、8003）

```bash
# 1. 装 Docker
curl -fsSL https://get.docker.com | sh
# 2. 拿代码，填密钥
git clone <本仓库> && cd dh_live/lulu-backend
cp .env.example .env && nano .env          # 填 ELEVENLABS_API_KEY、ANTHROPIC_API_KEY 等
# 3. 改配置里的服务器 IP 和噜噜音色
nano data/config.template.yaml             # <服务器IP>、<噜噜音色的 voice_id>
# 4. 启动
docker compose up -d --build
docker compose logs -f xiaozhi-server      # 看到 OTA 地址和 websocket 地址即成功
# 5. 浏览器打开 http://<服务器IP>:8003/xiaozhi/ota/ 能看到“OTA接口运行正常”
```

固件那边把 OTA 地址改成 `http://<服务器IP>:8003/xiaozhi/ota/`，见《自建后端替换计划》第 2 步。

## 视觉（拍照看图）

固件自带 `self.camera.take_photo` 工具，上游服务器自带 `/mcp/vision/explain` 接口，图片只在内存中转成 base64 交给模型，不落盘。我们只需要：

- `server.vision_explain` 填成公网地址；
- `VLLM` 选 `GatewayVLLM`，网关里 `lulu-vision` 接一个多模态模型（测试期云 API，自建期 Qwen3-VL-8B）。

## 回声消除 / 打断：现状和结论

- 上游服务器的“服务器端 AEC”只在 MQTT 网关接入时生效（16 字节包头带时间戳），websocket 直连不生效；算法是简单的谱减法。上游的打断逻辑是“播放时只要有一帧检测到人声就打断”，容易误打断。
- 我写了 AEC3（livekit 的 WebRTC AudioProcessingModule）+ 防误打断判定（播放开始 400ms 内不打断、累计 300ms 人声才打断），用 FLEURS 录音模拟“噜噜喇叭离麦克风很近、孩子插话”。结果见 `tests/bench_aec_bargein.py` 的输出：纯回声段只压下 6～36 dB，波动很大；12 个场景只有 1 个做到“孩子开口 1 秒内打断且不提前误打断”。
- 原因是服务器拿到的时间戳只有 60ms 粒度、还有 ±1 帧误差，参考信号和回声对不齐，AEC3 收敛不稳定。
- **建议改走设备端软件回采**：噜噜自己清楚每一刻往喇叭送了什么 PCM、时序是确定的，把这路 PCM 当参考信号送进 ESP-SR AFE 的 AEC（等同于给 `NoAudioCodec` 加软件 loopback），对齐精度比服务器端高一个量级。这是固件改动，需要真机调。服务器端保留防误打断判定，配合唤醒词 / 按键打断兜底。

## 高棉语评测

```bash
python eval/khmer_eval.py asr --parquet fleurs_km_test.parquet -n 30 --models whisper-turbo mms whisper-small-km wav2vec2-km
ELEVENLABS_API_KEY=... python eval/khmer_eval.py asr --parquet ... --models elevenlabs
python eval/khmer_eval.py tts --parquet ... -n 15 --tts mms-tts --judge <最好的 ASR>
```

FLEURS 下载：`huggingface_hub.hf_hub_download('google/fleurs', 'parquet-data/km_kh/test-00000-of-00001.parquet', repo_type='dataset')`。结果写在 `eval/results.jsonl`。
