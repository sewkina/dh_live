import re
import time
import os
from config.logger import setup_logging
from typing import Optional, Tuple, List
from core.providers.asr.dto.dto import InterfaceType
from core.providers.asr.base import ASRProviderBase

import requests

TAG = __name__
logger = setup_logging()

CJK_OR_KHMER_RE = re.compile(r"[\u4e00-\u9fff\u1780-\u17ff]")


class ASRProvider(ASRProviderBase):
    """ElevenLabs Scribe 语音识别，默认识别高棉语（language_code=khm）。"""

    def __init__(self, config: dict, delete_audio_file: bool):
        self.interface_type = InterfaceType.NON_STREAM
        self.api_key = config.get("api_key")
        self.api_url = config.get("base_url", "https://api.elevenlabs.io/v1/speech-to-text")
        self.model = config.get("model_name", "scribe_v1")
        self.language = config.get("language", "khm")
        self.output_dir = config.get("output_dir")
        self.delete_audio_file = delete_audio_file

        os.makedirs(self.output_dir, exist_ok=True)

    def requires_file(self) -> bool:
        return True

    async def speech_to_text(self, opus_data: List[bytes], session_id: str, artifacts=None) -> Tuple[Optional[str], Optional[str]]:
        file_path = None
        try:
            if artifacts is None:
                return "", None
            file_path = artifacts.file_path

            result = self._transcribe(file_path, self.language)
            text = result.get("text", "")
            # 自动判断语言时，孩子说的高棉语常被认成英文短词（如 “Sustain”“Did you?”）。
            # 结果既不是中文也不是高棉文、且语言把握不大或很短时，按高棉语再识别一次；
            # 重识别出高棉文才采用，真说英文的句子不受影响
            if not self.language and text and not CJK_OR_KHMER_RE.search(text):
                unsure = result.get("language_probability", 1.0) < 0.85 or len(text.split()) <= 3
                if unsure:
                    retry = self._transcribe(file_path, "khm").get("text", "")
                    if CJK_OR_KHMER_RE.search(retry):
                        logger.bind(tag=TAG).info(f"按高棉语重新识别: {text!r} -> {retry!r}")
                        text = retry
            return text, file_path

        except Exception as e:
            logger.bind(tag=TAG).error(f"语音识别失败: {e}")
            return "", None

    def _transcribe(self, file_path, language):
        headers = {"xi-api-key": self.api_key}
        data = {"model_id": self.model, "tag_audio_events": "false"}
        if language:
            data["language_code"] = language
        with open(file_path, "rb") as audio_file:
            start_time = time.time()
            response = requests.post(
                self.api_url, files={"file": audio_file}, data=data, headers=headers, timeout=30
            )
            logger.bind(tag=TAG).debug(
                f"语音识别耗时: {time.time() - start_time:.3f}s | 结果: {response.text}"
            )
        if response.status_code != 200:
            raise Exception(f"API请求失败: {response.status_code} - {response.text}")
        return response.json()
