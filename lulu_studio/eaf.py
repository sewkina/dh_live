"""噜噜屏幕表情 .eaf 文件的校验和预览（格式见 lulu_tools/emotes/make_emotes.py）。"""

import io
import struct


def check_eaf(data: bytes) -> str:
    """返回空字符串表示没问题，否则是错误说明。"""
    if len(data) < 16 or data[:4] != b"\x89EAF":
        return "不是噜噜表情文件（.eaf 文件头不对）"
    total, checksum, table_len = struct.unpack_from("<III", data, 4)
    if table_len != len(data) - 16:
        return "文件长度和文件头记录的不一致，可能没传完整"
    if checksum != sum(data[16:]) & 0xFFFFFFFF:
        return "校验和不对，文件可能损坏"
    if total < 1 or 16 + total * 8 > len(data):
        return "帧表不对"
    return ""


def decode_frames(data: bytes, max_frames: int = 200):
    """把 .eaf 解成 Pillow 图片列表（需要 Pillow）。"""
    from PIL import Image

    total = struct.unpack_from("<I", data, 4)[0]
    base = 16 + total * 8
    frames = []
    for i in range(min(total, max_frames)):
        _size, off = struct.unpack_from("<II", data, 16 + i * 8)
        b = base + off + 2
        if data[b:b + 2] != b"_S":
            continue
        w, h = struct.unpack_from("<HH", data, b + 10)
        nb, _bh = struct.unpack_from("<HH", data, b + 14)
        lens = struct.unpack_from("<%dI" % nb, data, b + 18)
        pal_off = b + 18 + nb * 4
        pal = []
        for k in range(256):
            pal += [data[pal_off + k * 4 + 2], data[pal_off + k * 4 + 1], data[pal_off + k * 4]]
        cur = pal_off + 1024
        px = bytearray()
        for n in lens:
            blk = data[cur + 1:cur + n]
            for k in range(0, len(blk) - 1, 2):
                px += bytes([blk[k + 1]]) * blk[k]
            cur += n
        if len(px) != w * h:
            continue
        im = Image.frombytes("P", (w, h), bytes(px))
        im.putpalette(pal)
        frames.append(im.convert("RGB"))
    return frames


def eaf_to_gif(data: bytes, fps: int = 10) -> bytes:
    frames = decode_frames(data)
    if not frames:
        raise ValueError("没解出画面")
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=max(20, 1000 // max(1, fps)),
                   loop=0, disposal=1)
    return buf.getvalue()
