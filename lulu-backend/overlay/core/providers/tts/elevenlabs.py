import requests
from core.utils.util import check_model_key
from core.providers.tts.base import TTSProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class TTSProvider(TTSProviderBase):
    """ElevenLabs 语音合成。voice_id 填“声音设计”或克隆出来的噜噜音色。"""

    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.api_key = config.get("api_key")
        self.voice = config.get("private_voice") or config.get("voice_id")
        self.model = config.get("model_id", "eleven_v3")
        self.api_url = config.get("api_url", "https://api.elevenlabs.io/v1/text-to-speech")
        self.audio_file_type = "mp3"
        self.output_format = config.get("output_format", "mp3_44100_128")
        self.language = config.get("language_code", "")
        self.output_file = config.get("output_dir", "tmp/")
        model_key_msg = check_model_key("TTS", self.api_key)
        if model_key_msg:
            logger.bind(tag=TAG).error(model_key_msg)

    async def text_to_speak(self, text, output_file):
        body = {"text": text, "model_id": self.model}
        if self.language:
            body["language_code"] = self.language
        response = requests.post(
            f"{self.api_url}/{self.voice}",
            params={"output_format": self.output_format},
            json=body,
            headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=self.tts_timeout,
        )
        if response.status_code != 200:
            raise Exception(f"ElevenLabs TTS请求失败: {response.status_code} - {response.text}")
        if output_file:
            with open(output_file, "wb") as f:
                f.write(response.content)
        else:
            return response.content
