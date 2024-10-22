import os
import numpy as np
import cv2
import gradio as gr
from gtts import gTTS
import soundfile as sf
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip

# 初始化模型
audioModel = AudioModel()
audioModel.loadModel("checkpoint/audio.pkl")

renderModel = RenderModel()
renderModel.loadModel("checkpoint/render.pth")
test_video = "ceshi"
pkl_path = f"video_data/{test_video}/keypoint_rotate.pkl"
video_path = f"video_data/{test_video}/circle.mp4"
renderModel.reset_charactor(video_path, pkl_path)


def text_to_video(text):
    # 生成语音
    tts = gTTS(text=text, lang='zh-cn')
    tts.save("temp.mp3")

    # 将MP3转换为WAV并读取
    os.system("ffmpeg -i temp.mp3 -acodec pcm_s16le -ar 16000 temp.wav")
    audio, _ = sf.read("temp.wav")

    frames = []
    sample_rate = 16000
    samples_per_read = int(0.04 * sample_rate)

    for i in range(0, len(audio), samples_per_read):
        pcm_data = audio[i:i + samples_per_read]
        if len(pcm_data) < samples_per_read:
            pcm_data = np.pad(pcm_data, (0, samples_per_read - len(pcm_data)))

        mouth_frame = audioModel.interface_frame(pcm_data)
        frame = renderModel.interface(mouth_frame)
        frames.append(frame)

    # 创建视频文件（仅视频，无音频）
    temp_video_file = "temp_video.mp4"
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_file, fourcc, 25, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()

    # 使用moviepy合并视频和音频
    video = VideoFileClip(temp_video_file)
    audio = AudioFileClip("temp.mp3")
    final_clip = video.set_audio(audio)

    output_file = "output_video.mp4"
    final_clip.write_videofile(output_file, codec='libx264', audio_codec='aac')

    # 清理临时文件
    os.remove("temp.mp3")
    os.remove("temp.wav")
    os.remove(temp_video_file)

    return output_file


# 创建Gradio接口
iface = gr.Interface(
    fn=text_to_video,
    inputs=gr.Textbox(lines=3, placeholder="输入要转换的文字..."),
    outputs=gr.Video(),
    title="数字人 TTS 系统",
    description="输入文字，生成数字人说话的视频（带声音）。"
)

# 启动Gradio应用
iface.launch()