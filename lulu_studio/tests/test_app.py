import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
os.environ["LULU_STUDIO_DATA"] = tempfile.mkdtemp()
os.environ["LULU_STUDIO_TOKEN"] = "t0ken"
sys.path.insert(0, str(HERE))

from fastapi.testclient import TestClient  # noqa: E402

import app as studio  # noqa: E402
from eaf import check_eaf, eaf_to_gif  # noqa: E402

c = TestClient(studio.app)
H = {"X-Lulu-Token": "t0ken"}
EAF = HERE / "seed/emotes/white_happy/happy.eaf"
GOOD = {"name": "spin_test", "interp": "spline", "frames": [{"d": 300, "p": [-600, 600, -600, 600, 200]}],
        "loop": {"from": 0, "to": 0, "times": 2}}


def test_seed_lists():
    for kind, n in (("actions", 11), ("emotes", 36), ("voices", 4)):
        r = c.get("/api/items", params={"type": kind})
        assert r.status_code == 200 and len(r.json()) >= n


def test_seed_actions_valid():
    for it in c.get("/api/items", params={"type": "actions"}).json():
        studio.validate_action(json.dumps(it["action"]).encode())


def test_seed_eaf_valid_and_decodes():
    for it in c.get("/api/items", params={"type": "emotes"}).json():
        data = c.get(it["main"]).content
        assert check_eaf(data) == "", it["id"]
    assert eaf_to_gif(EAF.read_bytes())[:3] == b"GIF"


def test_upload_needs_token():
    r = c.post("/api/upload", data={"type": "actions", "name": "x"},
               files={"main": ("a.json", json.dumps(GOOD), "application/json")})
    assert r.status_code == 401


def test_upload_action_download_delete():
    r = c.post("/api/upload", headers=H, data={"type": "actions", "name": "转圈圈", "tags": "测试 好玩"},
               files={"main": ("a.json", json.dumps(GOOD), "application/json")})
    assert r.status_code == 200, r.text
    it = r.json()
    assert it["source"] == "upload" and it["tags"] == ["测试", "好玩"]
    assert c.get(it["main"] + "?dl=1").json()["name"] == "spin_test"
    assert c.get(f"/api/items/actions/{it['id']}").json()["downloads"] == 1
    assert c.get(it["zip"]).content[:2] == b"PK"
    assert c.delete(f"/api/items/actions/{it['id']}").status_code == 401
    assert c.delete(f"/api/items/actions/{it['id']}", headers=H).status_code == 200
    assert c.get(f"/api/items/actions/{it['id']}").status_code == 404


def test_upload_action_rejects_bad():
    bad = dict(GOOD, frames=[{"d": 5, "p": [0, 0, 0, 0]}])
    r = c.post("/api/upload", headers=H, data={"type": "actions", "name": "x"},
               files={"main": ("a.json", json.dumps(bad), "application/json")})
    assert r.status_code == 400 and "p" in r.json()["error"]


def test_upload_emote_generates_preview():
    r = c.post("/api/upload", headers=H, data={"type": "emotes", "name": "我的开心", "emote": "happy"},
               files={"main": ("happy.eaf", EAF.read_bytes(), "application/octet-stream")})
    assert r.status_code == 200, r.text
    it = r.json()
    assert it["preview"].endswith("preview.gif") and c.get(it["preview"]).content[:3] == b"GIF"
    broken = bytearray(EAF.read_bytes())
    broken[100] ^= 0xFF
    r = c.post("/api/upload", headers=H, data={"type": "emotes", "name": "坏的"},
               files={"main": ("x.eaf", bytes(broken), "application/octet-stream")})
    assert r.status_code == 400


def test_upload_voice():
    r = c.post("/api/upload", headers=H, data={"type": "voices", "name": "奶声", "sample_text": "សួស្តី"},
               files={"sound": ("s.mp3", b"ID3fake", "audio/mpeg")})
    assert r.status_code == 200, r.text
    assert r.json()["sample_text"] == "សួស្តី"


def test_no_path_escape():
    assert c.get("/files/actions/wave_demo/meta.json").status_code == 404
    assert c.get("/files/actions/..%2F..%2Fapp.py/x").status_code == 404
    assert c.get("/api/items/actions/../x").status_code == 404


def test_static_index():
    r = c.get("/")
    assert r.status_code == 200 and "噜噜工坊" in r.text


def test_pack_subset():
    import io, zipfile
    r = c.get("/api/pack/emotes.zip", params={"ids": "dark_happy,dark_sad"})
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert "dark_happy/happy.eaf" in names and not any(n.startswith("white_") for n in names)
    assert len(json.loads(z.read("emote_list.json"))) == 2
