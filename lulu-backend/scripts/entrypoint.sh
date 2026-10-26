#!/bin/sh
# 把模板里的 ${VAR} 换成环境变量，生成真正的配置文件（密钥只存在于容器环境里）
python - <<'PY'
import os
src = "data/config.template.yaml"
if os.path.exists(src):
    with open(src, encoding="utf-8") as f:
        text = os.path.expandvars(f.read())
    with open("data/.config.yaml", "w", encoding="utf-8") as f:
        f.write(text)
PY
exec "$@"
