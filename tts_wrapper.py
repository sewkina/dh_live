from gtts import gTTS
import os

class TTSWrapper:
    def __init__(self):
        pass

    def generate_audio(self, text, output_file, lang='zh-cn'):
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(output_file)
            return True
        except Exception as e:
            print(f"音频生成失败: {str(e)}")
            return False