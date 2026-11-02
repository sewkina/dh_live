# 用 ElevenLabs scribe_v1 回读算高棉语错字率；用法: python cer_scribe.py <cmp_out目录>
import os, sys, re, json, glob, unicodedata, asyncio, requests
D = sys.argv[1]
KEY = os.environ['ELEVENLABS_API_KEY']
SENTS = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sents_km.txt'), encoding='utf-8').read().strip().split('\n')
def norm(s): return re.sub(r'[\s\?\!\.,។៕!？！​]', '', unicodedata.normalize('NFC', s))
def cer(r, h):
    r, h = norm(r), norm(h); d = list(range(len(h)+1))
    for i, a in enumerate(r, 1):
        p, d[0] = d[0], i
        for j, b in enumerate(h, 1): p, d[j] = d[j], min(d[j]+1, d[j-1]+1, p+(a!=b))
    return d[len(h)]/max(len(r),1)
def stt(path):
    with open(path,'rb') as f:
        r = requests.post('https://api.elevenlabs.io/v1/speech-to-text', headers={'xi-api-key': KEY},
                          data={'model_id':'scribe_v1','language_code':'khm'}, files={'file': f}, timeout=120)
    r.raise_for_status(); return r.json().get('text','')
async def edge():
    import edge_tts
    for i, s in enumerate(SENTS, 1):
        p = f'{D}/edge_{i:02d}.mp3'
        if not os.path.exists(p): await edge_tts.Communicate(s, 'km-KH-SreymomNeural').save(p)
try: asyncio.run(edge())
except Exception as e: print('Edge 失败:', e)
res = {}
for tag in ['base', 'base_clone', 's3000', 's3000_clone', 's2000', 's2000_clone', 'edge']:
    ext = 'mp3' if tag == 'edge' else 'wav'
    fs = [f'{D}/{tag}_{i:02d}.{ext}' for i in range(1, len(SENTS)+1)]
    if not all(os.path.exists(f) for f in fs): continue
    hyps = [stt(f) for f in fs]; cs = [cer(r, h) for r, h in zip(SENTS, hyps)]
    res[tag] = {'cer': sum(cs)/len(cs), 'cers': cs, 'hyps': hyps}
    print(f'{tag:12s} 平均错字率 {res[tag]["cer"]:.1%}  ' + ' '.join(f'{c:.0%}' for c in cs), flush=True)
json.dump(res, open(f'{D}/result_scribe.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
