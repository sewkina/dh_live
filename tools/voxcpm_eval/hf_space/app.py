# 噜噜高棉语 TTS（VoxCPM2 全量微调 3000 步，bf16）— 私有 ZeroGPU Space
import os, numpy as np, torch, gradio as gr, spaces
from huggingface_hub import snapshot_download
from voxcpm.core import VoxCPM

MODEL_REPO = os.environ.get('MODEL_REPO', 'jing96963/lulu-voxcpm2-khmer')
mdir = snapshot_download(MODEL_REPO, token=os.environ.get('HF_TOKEN'))
model = VoxCPM.from_pretrained(hf_model_id=mdir, load_denoiser=False, optimize=False)
PROMPT_WAV = os.path.join(mdir, 'prompts', 'lulu_km_prompt.wav')
PROMPT_TEXT = open(os.path.join(mdir, 'prompts', 'lulu_km_prompt.txt'), encoding='utf-8').read().strip()
REF_WAV = os.path.join(mdir, 'prompts', 'lulu_ref_km.wav')
CUTE_DESC = '(可爱活泼的小女孩，声音甜美、语调上扬)'

@spaces.GPU(duration=60)
def tts(text: str, mode: str = 'prompt', seed: int = 42):
    """mode: cute=可爱女声描述（推荐）/ prompt=续写噜噜音色 / ref=参考音克隆 / plain=默认声音"""
    torch.manual_seed(int(seed))
    kw = {}
    if mode == 'cute':
        text = CUTE_DESC + text
    if mode == 'prompt':
        kw = dict(prompt_wav_path=PROMPT_WAV, prompt_text=PROMPT_TEXT)
    elif mode == 'ref':
        kw = dict(reference_wav_path=REF_WAV)
    wav = model.generate(text=text, **kw)
    return model.tts_model.sample_rate, (np.clip(wav, -1, 1) * 32767).astype(np.int16)

demo = gr.Interface(fn=tts,
    inputs=[gr.Textbox(label='文本'), gr.Radio(['cute', 'prompt', 'ref', 'plain'], value='cute', label='模式'), gr.Number(42, label='seed')],
    outputs=gr.Audio(label='音频'), title='噜噜高棉语 TTS（私有测试）', api_name='tts')
demo.queue(default_concurrency_limit=4).launch()
