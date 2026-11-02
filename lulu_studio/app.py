"""噜噜工坊 · 资源库

浏览、试看试听、下载噜噜的动作、表情、音色，也可以上传自己做的。
同一个网页在浏览器和 Telegram 小程序里都能用。

启动：
    pip install -r requirements.txt
    python app.py                 # 默认 http://0.0.0.0:8020

环境变量：
    LULU_STUDIO_PORT   端口，默认 8020
    LULU_STUDIO_DATA   上传内容和下载次数存放目录，默认 ./data
    LULU_STUDIO_TOKEN  上传/删除口令。开了公网隧道（给 Telegram 用）时一定要设
"""

import io
import json
import os
import re
import secrets
import threading
import time
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from eaf import check_eaf, eaf_to_gif

ROOT = Path(__file__).resolve().parent
SEED = ROOT / "seed"
DATA = Path(os.environ.get("LULU_STUDIO_DATA", ROOT / "data")).resolve()
UPLOADS = DATA / "library"
STATS = DATA / "stats.json"
TOKEN = os.environ.get("LULU_STUDIO_TOKEN", "")

TYPES = {"actions": "动作", "emotes": "表情", "voices": "音色"}
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
FILE_RE = re.compile(r"^[A-Za-z0-9_\-.]{1,80}$")
AUDIO_EXT = {".mp3", ".ogg", ".wav", ".m4a", ".opus"}
LIMITS = {"action": 64 * 1024, "eaf": 8 * 1024 * 1024, "gif": 8 * 1024 * 1024, "sfx": 1024 * 1024,
          "voice": 10 * 1024 * 1024, "preset": 64 * 1024}

app = FastAPI(title="噜噜工坊")
_lock = threading.Lock()


# ---------------------------------------------------------------- 资源读取

def _roots():
    return [("official", SEED), ("upload", UPLOADS)]


def _item_dir(kind, item_id):
    if kind not in TYPES or not ID_RE.match(item_id):
        raise HTTPException(404, "没有这个资源")
    for source, base in _roots():
        d = base / kind / item_id
        if (d / "meta.json").is_file():
            return source, d
    raise HTTPException(404, "没有这个资源")


def _load_stats():
    try:
        return json.loads(STATS.read_text("utf-8"))
    except (OSError, ValueError):
        return {}


def _bump(kind, item_id):
    with _lock:
        s = _load_stats()
        key = f"{kind}/{item_id}"
        s[key] = s.get(key, 0) + 1
        DATA.mkdir(parents=True, exist_ok=True)
        tmp = STATS.with_suffix(".tmp")
        tmp.write_text(json.dumps(s, ensure_ascii=False), "utf-8")
        tmp.replace(STATS)


def _describe(kind, source, d, stats):
    meta = json.loads((d / "meta.json").read_text("utf-8"))
    item_id = d.name
    base = f"/files/{kind}/{item_id}/"
    files = sorted(p.name for p in d.iterdir() if p.is_file() and p.name != "meta.json")
    out = {
        "id": item_id,
        "type": kind,
        "source": source,
        "name": meta.get("name") or item_id,
        "desc": meta.get("desc", ""),
        "tags": meta.get("tags", []),
        "author": meta.get("author", ""),
        "created": meta.get("created", 0),
        "order": meta.get("order", 1000),
        "downloads": stats.get(f"{kind}/{item_id}", 0),
        "files": files,
        "size": sum((d / f).stat().st_size for f in files),
        "zip": f"/api/items/{kind}/{item_id}/zip",
    }
    for key in ("main", "preview", "sound"):
        if meta.get(key) and (d / meta[key]).is_file():
            out[key] = base + meta[key]
    for key in ("emote", "fps", "loop", "sample_text"):
        if key in meta:
            out[key] = meta[key]
    if kind == "actions" and meta.get("main"):
        try:
            out["action"] = json.loads((d / meta["main"]).read_text("utf-8"))
        except (OSError, ValueError):
            pass
    return out


def list_items(kind=None):
    stats = _load_stats()
    items = []
    for k in ([kind] if kind else TYPES):
        for source, base in _roots():
            folder = base / k
            if not folder.is_dir():
                continue
            for d in sorted(folder.iterdir()):
                if (d / "meta.json").is_file():
                    try:
                        items.append(_describe(k, source, d, stats))
                    except (OSError, ValueError):
                        continue
    # 官方的按设定顺序，上传的新的在前
    items.sort(key=lambda i: (i["type"], i["source"] != "official", i["order"], -i["created"], i["id"]))
    return items


# ---------------------------------------------------------------- 校验

def validate_action(raw: bytes):
    """和固件关键帧播放器的限制一致：最多 40 帧、JSON 小于 4000 字节、5 个关节。"""
    try:
        a = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(400, "动作文件不是合法的 JSON")
    if not isinstance(a, dict):
        raise HTTPException(400, "动作 JSON 顶层应该是一个对象")
    name = a.get("name", "")
    if not isinstance(name, str) or not re.match(r"^[A-Za-z0-9_]{1,24}$", name):
        raise HTTPException(400, "动作 name 只能用英文字母、数字、下划线，最长 24 个字符")
    frames = a.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 40:
        raise HTTPException(400, "frames 需要 1~40 帧")
    for i, f in enumerate(frames, 1):
        p, d = (f or {}).get("p"), (f or {}).get("d")
        if not isinstance(p, list) or len(p) != 5 or not all(isinstance(v, int) and -3000 <= v <= 3000 for v in p):
            raise HTTPException(400, f"第 {i} 帧的 p 要是 5 个 -3000~3000 的整数（左前、右前、左后、右后、腰）")
        if not isinstance(d, int) or not 20 <= d <= 10000:
            raise HTTPException(400, f"第 {i} 帧的 d 要是 20~10000 毫秒")
    loop = a.get("loop")
    if loop is not None:
        ok = isinstance(loop, dict) and all(isinstance(loop.get(k), int) for k in ("from", "to", "times"))
        if not ok or not 0 <= loop["from"] <= loop["to"] < len(frames) or not 1 <= loop["times"] <= 50:
            raise HTTPException(400, "loop 要写成 {\"from\":起始帧,\"to\":结束帧,\"times\":次数}，帧号从 0 开始")
    compact = json.dumps(a, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(compact) >= 4000:
        raise HTTPException(400, "动作 JSON 超过 4000 字节，噜噜存不下，请减少帧数")
    return a


async def _read(up: UploadFile, limit, what):
    data = await up.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(400, f"{what}太大了（上限 {limit // 1024} KB）")
    if not data:
        raise HTTPException(400, f"{what}是空的")
    return data


def _ext(up: UploadFile):
    return Path(up.filename or "").suffix.lower()


def _check_token(token):
    if TOKEN and not secrets.compare_digest(token or "", TOKEN):
        raise HTTPException(401, "口令不对。上传和删除需要口令，请在页面右上角“设置”里填写")


def _clean_text(s, n):
    return re.sub(r"[\x00-\x1f\x7f]", " ", (s or "").strip())[:n]


# ---------------------------------------------------------------- 接口

@app.get("/api/config")
def api_config():
    return {"types": TYPES, "upload_needs_token": bool(TOKEN)}


@app.get("/api/items")
def api_items(type: str = Query(None), q: str = Query("")):
    if type and type not in TYPES:
        raise HTTPException(400, "type 只能是 actions / emotes / voices")
    items = list_items(type)
    if q:
        ql = q.lower()
        items = [i for i in items if ql in (i["name"] + i["desc"] + i["id"] + " ".join(i["tags"]) + i["author"]).lower()]
    return items


@app.get("/api/items/{kind}/{item_id}")
def api_item(kind: str, item_id: str):
    source, d = _item_dir(kind, item_id)
    return _describe(kind, source, d, _load_stats())


@app.get("/files/{kind}/{item_id}/{fname}")
def api_file(kind: str, item_id: str, fname: str, dl: int = 0):
    _source, d = _item_dir(kind, item_id)
    if not FILE_RE.match(fname) or fname.startswith(".") or fname == "meta.json":
        raise HTTPException(404, "没有这个文件")
    p = d / fname
    if not p.is_file():
        raise HTTPException(404, "没有这个文件")
    if dl:
        _bump(kind, item_id)
        return FileResponse(p, filename=fname)
    return FileResponse(p)


@app.get("/api/items/{kind}/{item_id}/zip")
def api_zip(kind: str, item_id: str):
    _source, d = _item_dir(kind, item_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.iterdir()):
            if p.is_file():
                z.write(p, f"{item_id}/{p.name}")
    _bump(kind, item_id)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="lulu_{kind}_{item_id}.zip"'})


@app.get("/api/pack/{kind}.zip")
def api_pack(kind: str, ids: str = Query("")):
    """打包下载。ids 用逗号分隔，不填就是这一类全部。表情包里附带 emote_list.json（固件资源用的表情清单）。"""
    if kind not in TYPES:
        raise HTTPException(404, "没有这个分类")
    want = set(filter(None, ids.split(",")))
    items = [i for i in list_items(kind) if not want or i["id"] in want]
    if not items:
        raise HTTPException(404, "没有可打包的资源")
    buf = io.BytesIO()
    emote_list = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for it in items:
            _src, d = _item_dir(kind, it["id"])
            for p in sorted(d.iterdir()):
                if p.is_file():
                    z.write(p, f"{it['id']}/{p.name}")
            if kind == "emotes" and it.get("main"):
                emote_list.append({"emote": it.get("emote") or it["id"], "src": it["main"].rsplit("/", 1)[-1],
                                   "loop": it.get("loop", True), "fps": it.get("fps", 10)})
            _bump(kind, it["id"])
        if emote_list:
            z.writestr("emote_list.json", json.dumps(emote_list, ensure_ascii=False, indent=1))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="lulu_{kind}_pack.zip"'})


@app.post("/api/upload")
async def api_upload(
    type: str = Form(...),
    name: str = Form(...),
    desc: str = Form(""),
    author: str = Form(""),
    tags: str = Form(""),
    emote: str = Form(""),
    fps: int = Form(10),
    sample_text: str = Form(""),
    main: UploadFile = File(None),
    preview: UploadFile = File(None),
    sound: UploadFile = File(None),
    x_lulu_token: str = Header(""),
):
    _check_token(x_lulu_token)
    if type not in TYPES:
        raise HTTPException(400, "type 只能是 actions / emotes / voices")
    name = _clean_text(name, 40)
    if not name:
        raise HTTPException(400, "请填一个名字")
    meta = {"name": name, "desc": _clean_text(desc, 300), "author": _clean_text(author, 40) or "匿名",
            "tags": [t for t in (_clean_text(x, 16) for x in re.split(r"[,，\s]+", tags)) if t][:6],
            "created": int(time.time())}
    files = {}

    if type == "actions":
        if not main or _ext(main) != ".json":
            raise HTTPException(400, "动作请上传 .json 文件（和固件关键帧播放器同一种格式）")
        raw = await _read(main, LIMITS["action"], "动作文件")
        a = validate_action(raw)
        fname = a["name"] + ".json"
        files[fname] = json.dumps(a, ensure_ascii=False, indent=1).encode("utf-8")
        meta["main"] = fname
        slug = a["name"]
    elif type == "emotes":
        if not main or _ext(main) != ".eaf":
            raise HTTPException(400, "表情请上传噜噜屏幕用的 .eaf 文件")
        raw = await _read(main, LIMITS["eaf"], "表情文件")
        err = check_eaf(raw)
        if err:
            raise HTTPException(400, err)
        key = re.sub(r"[^a-z0-9_]", "", (emote or Path(main.filename).stem).lower())[:24] or "custom"
        meta.update(main=key + ".eaf", emote=key, fps=max(1, min(30, fps)), loop=True)
        files[key + ".eaf"] = raw
        if preview and preview.filename:
            if _ext(preview) not in (".gif", ".png", ".webp"):
                raise HTTPException(400, "预览图请用 .gif / .png / .webp")
            files["preview" + _ext(preview)] = await _read(preview, LIMITS["gif"], "预览图")
            meta["preview"] = "preview" + _ext(preview)
        else:
            try:
                files["preview.gif"] = eaf_to_gif(raw, meta["fps"])
                meta["preview"] = "preview.gif"
            except Exception:  # noqa: BLE001  没装 Pillow 或解不出来就不放预览
                pass
        if sound and sound.filename:
            if _ext(sound) not in AUDIO_EXT:
                raise HTTPException(400, "音效请用 mp3 / ogg / wav")
            files[f"sfx_{key}{_ext(sound)}"] = await _read(sound, LIMITS["sfx"], "音效")
            meta["sound"] = f"sfx_{key}{_ext(sound)}"
        slug = key
    else:
        if not sound or _ext(sound) not in AUDIO_EXT:
            raise HTTPException(400, "音色请上传一段试听音频（mp3 / ogg / wav）")
        files["sample" + _ext(sound)] = await _read(sound, LIMITS["voice"], "试听音频")
        meta["sound"] = "sample" + _ext(sound)
        preset = {"provider": "", "config": {}, "sample_text": _clean_text(sample_text, 300)}
        if main and main.filename:
            raw = await _read(main, LIMITS["preset"], "音色配置")
            try:
                preset = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise HTTPException(400, "音色配置要是 JSON 文件")
            if not isinstance(preset, dict):
                raise HTTPException(400, "音色配置 JSON 顶层应该是一个对象")
        files["preset.json"] = json.dumps(preset, ensure_ascii=False, indent=1).encode("utf-8")
        meta["main"] = "preset.json"
        if preset.get("sample_text") or sample_text:
            meta["sample_text"] = _clean_text(sample_text or str(preset.get("sample_text", "")), 300)
        slug = "voice"

    item_id = f"u{int(time.time())}_{secrets.token_hex(2)}_{slug}"[:64]
    d = UPLOADS / type / item_id
    d.mkdir(parents=True, exist_ok=False)
    for fname, data in files.items():
        (d / fname).write_bytes(data)
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf-8")
    return _describe(type, "upload", d, _load_stats())


@app.delete("/api/items/{kind}/{item_id}")
def api_delete(kind: str, item_id: str, x_lulu_token: str = Header("")):
    _check_token(x_lulu_token)
    source, d = _item_dir(kind, item_id)
    if source != "upload":
        raise HTTPException(403, "官方资源不能在网页上删除")
    trash = DATA / "trash" / kind
    trash.mkdir(parents=True, exist_ok=True)
    d.replace(trash / f"{item_id}_{int(time.time())}")  # 先挪进回收站，不直接删
    return {"ok": True}


@app.exception_handler(HTTPException)
async def _http_error(_req, exc):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("LULU_STUDIO_HOST", "0.0.0.0"),
                port=int(os.environ.get("LULU_STUDIO_PORT", "8020")))
