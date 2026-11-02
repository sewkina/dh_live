# 本地（WSL, conda env com）对比生成：原模型 / 微调 2000 / 3000 步，默认声音 + 噜噜克隆声音
# 用法: PYTORCH_JIT=0 python local_cmp.py   （只生成 wav，错字率在云端算）
import os, sys, shutil, gc, torch, soundfile as sf
MODEL_DIR = '/mnt/ai/ComfyUI/models/tts/VoxCPM/VoxCPM2'
CK = '/mnt/d/Xgo/voxcpm_ckpt'
REF = '/mnt/d/Xgo/dh_live/lulu-backend/voxcpm/lulu_ref.wav'
OUT = f'{CK}/cmp_out'; os.makedirs(OUT, exist_ok=True)
SENTS = '''សួស្តី! តើអ្នកឈ្មោះអ្វី?
ឆ្កែមានជើងប៉ុន្មាន?
មេឃមានពណ៌អ្វី?
មួយបូកពីរស្មើប៉ុន្មាន?
ហេតុអ្វីបានជាយើងត្រូវដុសធ្មេញ?
ខ្ញុំខ្លាចងងឹត។
សូមនិទានរឿងខ្លីមួយអំពីទន្សាយ។
ផ្លែស្វាយមានរសជាតិយ៉ាងម៉េច?
សូមរាប់ពីមួយដល់ប្រាំ។
ថ្ងៃនេះអាកាសធាតុល្អណាស់។
ប្អូនពូកែណាស់!
តោះយើងលេងជាមួយគ្នា។
ញ៉ាំបាយហើយឬនៅ?
ដល់ម៉ោងគេងហើយ រាត្រីសួស្តី។
ខ្ញុំស្រលាញ់ប្អូនណាស់។
ឆ្មាស្រែកថា មៀវ មៀវ។
ផ្កានេះមានពណ៌ក្រហម។
កុំយំអី ខ្ញុំនៅទីនេះ។
ខ្ញុំមានអាយុប្រាំឆ្នាំ។
លាហើយ ជួបគ្នាថ្ងៃស្អែក!'''.split('\n')
ZH = ['你好呀，我是噜噜，今天想玩什么游戏？', '小朋友真棒，我们一起数到十吧！', '天黑了，该睡觉啦，晚安。']
from voxcpm.core import VoxCPM
def load(step):
    if step == 0:
        return VoxCPM.from_pretrained(hf_model_id=MODEL_DIR, load_denoiser=False, optimize=False)
    d = f'{CK}/model_s{step}'; os.makedirs(d, exist_ok=True)
    for fn in os.listdir(MODEL_DIR):
        p = f'{d}/{fn}'
        if fn != 'model.safetensors' and not os.path.lexists(p): os.symlink(f'{MODEL_DIR}/{fn}', p)
    p = f'{d}/model.safetensors'
    if not os.path.lexists(p): os.symlink(f'{CK}/step_{step:07d}/model.safetensors', p)
    return VoxCPM.from_pretrained(hf_model_id=d, load_denoiser=False, optimize=False)
steps = [int(s) for s in sys.argv[1:]] or [0, 2000, 3000]
for step in steps:
    tag0 = 'base' if step == 0 else f's{step}'
    m = load(step)
    for ref, sfx in [(REF, '_clone'), (None, '')]:
        tag = tag0 + sfx
        if os.path.exists(f'{OUT}/zh_{tag}_3.wav'): print('跳过', tag); continue
        for i, s in enumerate(SENTS, 1):
            sf.write(f'{OUT}/{tag}_{i:02d}.wav', m.generate(text=s, reference_wav_path=ref, seed=42), m.tts_model.sample_rate)
        for i, t in enumerate(ZH, 1):
            sf.write(f'{OUT}/zh_{tag}_{i}.wav', m.generate(text=t, reference_wav_path=ref, seed=42), m.tts_model.sample_rate)
        print('生成完毕', tag, flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()
print('全部完成', OUT)
