import io
import re

import edge_tts
import requests
from pydub import AudioSegment
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()

# 高棉文字（含高棉符号区）。片段之间的空格、标点归到相邻片段里
KHMER_CHARS = "ក-៿᧠-᧿"
SEGMENT_RE = re.compile(rf"[{KHMER_CHARS}]+(?:[\s​]*[{KHMER_CHARS}]+)*|[^{KHMER_CHARS}]+")
KHMER_RE = re.compile(rf"[{KHMER_CHARS}]")
SPEAKABLE_RE = re.compile(r"[\wក-៿]")


class TTSProvider(TTSProviderBase):
    """中文、英语用 ElevenLabs；高棉语用微软 Edge 语音。

    实测 ElevenLabs（含 eleven_v3）念高棉语，回读会被识别成泰语、韩语、英语乱码；
    Edge 的 km-KH-SreymomNeural / PisethNeural 回读几乎一字不差。
    一句话里混着两种文字时拆段分别合成，再拼成一段。
    注意：Edge 免费接口只适合测试，商用要换 Azure 语音（同款声音）。
    """

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.api_key = config.get("api_key")
        self.voice = config.get("private_voice") or config.get("voice_id")
        self.model = config.get("model_id", "eleven_flash_v2_5")
        self.khmer_voice = config.get("khmer_voice", "km-KH-SreymomNeural")
        self.api_url = config.get("api_url", "https://api.elevenlabs.io/v1/text-to-speech")
        self.audio_file_type = "mp3"
        self.output_format = config.get("output_format", "mp3_44100_128")
        self.output_file = config.get("output_dir", "tmp/")
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

    def _elevenlabs(self, text):
        response = requests.post(
            f"{self.api_url}/{self.voice}",
            params={"output_format": self.output_format},
            json={"text": text, "model_id": self.model},
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=self.tts_timeout,
        )
        if response.status_code != 200:
            raise Exception(f"ElevenLabs TTS请求失败: {response.status_code} - {response.text}")
        return response.content

    async def _edge(self, text):
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, voice=self.khmer_voice).stream():
            if chunk["type"] == "audio":
                audio.extend(chunk["data"])
        if not audio:
            raise Exception("Edge 高棉语合成没有返回音频")
        return bytes(audio)

    async def _synth(self, text):
        segments = [s for s in SEGMENT_RE.findall(text) if SPEAKABLE_RE.search(s)]
        if not segments:
            return b""
        parts = []
        for segment in segments:
            if KHMER_RE.search(segment):
                parts.append(await self._edge(segment))
            else:
                parts.append(self._elevenlabs(segment))
        if len(parts) == 1:
            return parts[0]
        merged = AudioSegment.empty()
        for part in parts:
            merged += AudioSegment.from_file(io.BytesIO(part), format="mp3")
        out = io.BytesIO()
        merged.export(out, format="mp3")
        return out.getvalue()

    async def text_to_speak(self, text, output_file):
        audio = await self._synth(text)
        if output_file:
            with open(output_file, "wb") as f:
                f.write(audio)
        else:
            return audio
