#!/bin/sh
# 备选：用 SGLang 代替 vLLM 起自建模型（在 GPU 服务器上直接跑）
# 起来后把 gateway/litellm.yaml 里的 api_base 改成 http://<这台机器>:30000/v1
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
docker run -d --gpus all --name lulu-sglang -p 30000:30000 \
  -v "$PWD/models-cache:/root/.cache/huggingface" --ipc=host \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name local-model \
    --host 0.0.0.0 --port 30000 --api-key "${SGLANG_API_KEY:-change-me}"
