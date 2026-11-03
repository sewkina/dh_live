"""噜噜“垫音”：孩子说完话、识别出文字后，马上先放一个很短的萌宠音效（“嗯~”之类），
再等大模型和 HF 合成的正式回答。让孩子知道噜噜听到了，不会以为它没反应。

- 音效放在 LULU_FILLER_DIR（默认 /opt/xiaozhi-esp32-server/lulu_filler），支持 ogg/wav/mp3，
  建议用已选定的表情音效（思考/好奇/在听），每段 0.3–0.8 秒。没有文件就什么都不做。
- 识别结果是“听不清”时也放（噜噜随后会请对方再说一遍）。
- 关掉：LULU_DISABLE_FILLER=1
"""

import asyncio
import glob
import inspect
import os
import random

TAG = "lulu_filler"
EXTS = ("*.ogg", "*.wav", "*.mp3", "*.opus")


def install():
    if os.environ.get("LULU_DISABLE_FILLER") == "1":
        return
    folder = os.environ.get("LULU_FILLER_DIR", "/opt/xiaozhi-esp32-server/lulu_filler")
    files = sorted(f for e in EXTS for f in glob.glob(os.path.join(folder, e)))
    if not files:
        print(f"lulu_filler: {folder} 里没有音效，跳过")
        return

    from core.handle import receiveAudioHandle as rah
    from core.handle import sendAudioHandle as sah
    from core.providers.tts.dto.dto import SentenceType
    from core.utils.util import audio_to_data

    if getattr(rah.startToChat, "_lulu_filler", False):
        return

    cache = {}

    async def load(path):
        if path not in cache:
            r = audio_to_data(path, is_opus=True)
            if inspect.isawaitable(r):
                r = await r
            cache[path] = r
        return cache[path]

    orig = rah.startToChat

    async def start_to_chat(conn, text, *a, **kw):
        try:
            if not getattr(conn, "client_abort", False):
                audios = await load(random.choice(files))
                if audios:
                    await sah.sendAudioMessage(conn, SentenceType.FIRST, audios, None)
        except Exception as e:  # 垫音失败不影响正常回答
            conn.logger.bind(tag=TAG).warning(f"垫音失败: {e}")
        return await orig(conn, text, *a, **kw)

    start_to_chat._lulu_filler = True
    rah.startToChat = start_to_chat
    # asr/base.py 是 from ... import startToChat，名字要一起换
    from core.providers.asr import base as asr_base
    if hasattr(asr_base, "startToChat"):
        asr_base.startToChat = start_to_chat
    print(f"lulu_filler: 已装上，{len(files)} 段音效")
