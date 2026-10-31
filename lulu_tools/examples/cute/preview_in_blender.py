"""在 Blender 里把这个文件夹的动作 JSON 依次排到时间轴上，按空格就能连着看（循环段会展开）。

用法：打开带 LuluRig 的文件（如插件生成的模型或 lulu_shell_preview.blend），
在“脚本”工作区打开本文件运行；或命令行：
  blender lulu_shell_preview.blend --python preview_in_blender.py -- 输出.blend
每个动作之间留 1 秒站立，时间轴上有动作名标记。
"""

import glob
import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "blender"))
from lulu_animator import lulu_link as L, rig  # noqa: E402

GAP_S = 1.0


def expand(action):
    """[(d_ms, angles)]，按 loop 把循环段展开。"""
    frames = L.action_frames(action)
    loop = action.get("loop")
    if not loop:
        return frames
    a, b, times = loop["from"], loop["to"], max(1, loop.get("times", 1))
    return frames[:b + 1] + frames[a:b + 1] * (times - 1) + frames[b + 1:]


def main(out=None):
    arm = rig.find_rig(bpy.context)
    if arm is None:
        arm = rig.build_rig(bpy.context)
    scene = bpy.context.scene
    fps = scene.render.fps / scene.render.fps_base
    if arm.animation_data and arm.animation_data.action:
        arm.animation_data.action = None
    for m in list(scene.timeline_markers):
        scene.timeline_markers.remove(m)
    t = 0.0
    rig.set_angles(arm, L.STAND_ANGLES)
    rig.keyframe_angles(arm, 1)
    for path in sorted(glob.glob(os.path.join(HERE, "*.json"))):
        with open(path, encoding="utf-8") as f:
            action = json.load(f)
        t += GAP_S * 1000
        scene.timeline_markers.new(action["name"], frame=1 + round(t * fps / 1000))
        rig.set_angles(arm, L.STAND_ANGLES)
        rig.keyframe_angles(arm, 1 + round(t * fps / 1000))
        for d, angles in expand(action):
            t += d
            rig.set_angles(arm, angles)
            rig.keyframe_angles(arm, 1 + round(t * fps / 1000))
    t += GAP_S * 1000
    end = 1 + round(t * fps / 1000)
    rig.set_angles(arm, L.STAND_ANGLES)
    rig.keyframe_angles(arm, end)
    scene.frame_start, scene.frame_end = 1, end
    scene.frame_set(1)
    print("已排好 %d 帧（%.1f 秒），时间轴上有动作名标记" % (end, t / 1000))
    if out:
        bpy.ops.wm.save_as_mainfile(filepath=out)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    main(argv[0] if argv else None)
