"""噜噜动作编辑器：在 Blender 里摆动作、实时操控真机、手掰录制、检查翻倒、导出动作 JSON。"""

bl_info = {
    "name": "Lulu Animator 噜噜动作编辑器",
    "author": "Jingim",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D 视图 > 侧边栏（N）> 噜噜",
    "description": "RIG-Puppy 噜噜的动作编辑、实时操控和导出",
    "category": "Animation",
}

import importlib
import json
import os
import subprocess
import sys
import time

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import lulu_link, rig

importlib.reload(lulu_link)
importlib.reload(rig)

from .lulu_link import JOINT_LABELS, JOINTS  # noqa: E402


class _State:
    link = None          # LuluLink
    last_send = 0.0
    sending = False      # 防止自己写骨骼触发的更新再发回去
    recording = False
    rec_start_frame = 0
    rec_frames = []      # [(frame, angles)]
    rec_joints = []
    status = "未连接"


S = _State()


def _modules_dir():
    path = bpy.utils.user_resource("SCRIPTS", path="modules", create=True)
    if path not in sys.path:
        sys.path.append(path)
    return path


def _deps_ok():
    _modules_dir()
    try:
        import serial  # noqa: F401
        import bleak  # noqa: F401
        return True
    except ImportError:
        return False


_PORT_ITEMS = []  # Blender 要求动态枚举的字符串保持引用


def _port_items(self, context):
    _PORT_ITEMS[:] = [(dev, "%s  %s" % (dev, desc), "") for dev, desc in lulu_link.list_serial_ports()]
    if not _PORT_ITEMS:
        _PORT_ITEMS.append(("NONE", "没有找到串口（先安装依赖或插上线）", ""))
    return _PORT_ITEMS


def _props(context):
    return context.scene.lulu


def _rig(context):
    return rig.find_rig(context)


# ------------------------------------------------------------------ 实时同步

def _send_current_pose(scene, force=False):
    link = S.link
    props = scene.lulu
    if not link or not props.live_sync or S.sending:
        return
    now = time.time()
    if not force and now - S.last_send < 1.0 / 40:
        return
    arm = rig.find_rig(bpy.context)
    if arm is None:
        return
    S.last_send = now
    try:
        link.send_angles(rig.get_angles(arm), ms=50)
    except Exception as e:  # noqa: BLE001
        S.status = "发送失败：%s" % e


@persistent
def _on_depsgraph(scene, depsgraph):
    _send_current_pose(scene)


@persistent
def _on_frame(scene, depsgraph=None):
    _send_current_pose(scene, force=True)


def _follow_timer():
    """跟随真机 / 录制：把真机实际姿态写回卸力的关节。"""
    context = bpy.context
    scene = context.scene
    props = scene.lulu
    link = S.link
    if not link or not (props.follow or S.recording):
        return None
    arm = rig.find_rig(context)
    if arm is None:
        return 0.1
    fps = scene.render.fps / scene.render.fps_base
    if S.recording:
        if scene.frame_current >= scene.frame_end:
            _stop_recording(context)
            return None
        scene.frame_set(scene.frame_current + 1)  # 上力的关节按已有动画播放（分轨）
    if link.pose:
        unloaded = [i for i in range(5) if not (link.torque_mask >> i) & 1]
        S.sending = True
        try:
            rig.set_angles(arm, link.pose, unloaded)
        finally:
            S.sending = False
        if S.recording:
            S.rec_frames.append((scene.frame_current, list(link.pose)))
    return 1.0 / fps if S.recording else 0.05


def _stop_recording(context):
    S.recording = False
    arm = _rig(context)
    props = _props(context)
    if not S.rec_frames or arm is None:
        return 0
    frames = [f for f, _ in S.rec_frames]
    poses = [p for _, p in S.rec_frames]
    fps = context.scene.render.fps / context.scene.render.fps_base
    times = [(f - frames[0]) * 1000.0 / fps for f in frames]
    keys, _eps = lulu_link.samples_to_frames(times, poses, epsilon=props.simplify_deg, max_frames=200,
                                             smooth_window=props.smooth_window)
    rig.delete_keys(arm, S.rec_joints, frames[0], frames[-1])
    t = 0.0
    for n, (d, angles) in enumerate(keys):
        if n:
            t += d
        f = frames[0] + round(t * fps / 1000.0)
        rig.set_angles(arm, angles, S.rec_joints)
        rig.keyframe_angles(arm, f, S.rec_joints)
    S.rec_frames = []
    S.status = "录制完成：%d 个关键帧" % len(keys)
    return len(keys)


# ------------------------------------------------------------------ 属性

def _torque_update(self, context):
    if S.link:
        mask = sum((1 << i) for i in range(5) if self.torque[i])
        ok, msg = S.link.set_torque(mask)
        S.status = "关节上力已更新" if ok else "上力设置失败：%s" % msg


def _follow_update(self, context):
    if S.link:
        S.link.stream(25 if (self.follow or S.recording) else 0)
        if self.follow and not bpy.app.timers.is_registered(_follow_timer):
            bpy.app.timers.register(_follow_timer)


class LuluProps(bpy.types.PropertyGroup):
    transport: EnumProperty(name="连接方式", items=[("USB", "USB 线", "用刷机的 USB-C 线（COMx）"),
                                                   ("BLE", "蓝牙", "噜噜需先开启蓝牙遥控模式")])
    port: EnumProperty(name="串口", items=_port_items)
    live_sync: BoolProperty(name="实时同步到真机", description="骨骼一动，真机跟着动", default=False)
    follow: BoolProperty(name="跟随真机", description="卸力的关节用手掰，骨骼跟着动", default=False,
                         update=_follow_update)
    torque: bpy.props.BoolVectorProperty(name="上力", size=5, default=(True,) * 5, update=_torque_update,
                                         description="取消勾选 = 卸力，可以用手掰")
    action_name: StringProperty(name="动作名", default="my_action")
    emotion: EnumProperty(name="表情", items=[(e, "不改表情" if e == "none" else e, "") for e in (
        "none", "happy", "laughing", "loving", "winking", "silly", "cool", "surprised", "sad", "sleepy",
        "thinking", "angry", "shocked", "kiss", "relaxed", "confident", "embarrassed", "crying")])
    interp: EnumProperty(name="插值", items=[("spline", "连续（样条）", "一气呵成，中途不停"),
                                           ("ease", "每帧停顿（缓动）", "每个姿势慢起慢停，适合点头等")])
    loop: BoolProperty(name="可重复", description="导出整个动作可循环，播放时可指定次数", default=False)
    simplify_deg: FloatProperty(name="精简误差（度）", default=1.0, min=0.2, max=10.0)
    smooth_window: IntProperty(name="去抖窗口", default=3, min=1, max=9)
    preview_speed: IntProperty(name="预览速度 %", default=100, min=25, max=400)


# ------------------------------------------------------------------ 操作

class LULU_OT_install_deps(bpy.types.Operator):
    bl_idname = "lulu.install_deps"
    bl_label = "安装依赖（pyserial、bleak）"

    def execute(self, context):
        target = _modules_dir()
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--target", target,
                                   "pyserial", "bleak"])
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, "安装失败：%s" % e)
            return {"CANCELLED"}
        self.report({"INFO"}, "依赖已安装")
        return {"FINISHED"}


class LULU_OT_build_rig(bpy.types.Operator):
    bl_idname = "lulu.build_rig"
    bl_label = "生成噜噜模型"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        rig.build_rig(context)
        return {"FINISHED"}


class LULU_OT_connect(bpy.types.Operator):
    bl_idname = "lulu.connect"
    bl_label = "连接噜噜"

    def execute(self, context):
        props = _props(context)
        if S.link:
            S.link.close()
            S.link = None
        try:
            S.link = lulu_link.LuluLink.usb(props.port) if props.transport == "USB" else lulu_link.LuluLink.ble()
        except Exception as e:  # noqa: BLE001
            S.status = "连接失败：%s" % e
            self.report({"ERROR"}, S.status)
            return {"CANCELLED"}
        version = S.link.ping()
        if not version:
            S.status = "已打开连接，但噜噜没有回应（固件是否已更新？）"
            self.report({"WARNING"}, S.status)
        else:
            S.status = "已连接（%s）" % version
        props.torque = (True,) * 5
        return {"FINISHED"}


class LULU_OT_disconnect(bpy.types.Operator):
    bl_idname = "lulu.disconnect"
    bl_label = "断开"

    def execute(self, context):
        props = _props(context)
        props.live_sync = False
        props.follow = False
        S.recording = False
        if S.link:
            S.link.close()
            S.link = None
        S.status = "未连接"
        return {"FINISHED"}


class LULU_OT_record(bpy.types.Operator):
    bl_idname = "lulu.record"
    bl_label = "开始录制卸力关节"
    bl_description = "从当前帧开始实时播放时间轴，上力的关节按已有动画动，卸力的关节记录你手掰的动作"

    def execute(self, context):
        if not S.link:
            self.report({"ERROR"}, "先连接噜噜")
            return {"CANCELLED"}
        if S.recording:
            n = _stop_recording(context)
            S.link.stream(25 if _props(context).follow else 0)
            self.report({"INFO"}, "录制完成：%d 个关键帧" % n)
            return {"FINISHED"}
        joints = [i for i in range(5) if not _props(context).torque[i]]
        if not joints:
            self.report({"ERROR"}, "先取消勾选要手掰的关节（卸力）")
            return {"CANCELLED"}
        S.rec_joints = joints
        S.rec_frames = []
        S.recording = True
        S.link.stream(25)
        _props(context).live_sync = True  # 让上力的关节跟着时间轴动
        if not bpy.app.timers.is_registered(_follow_timer):
            bpy.app.timers.register(_follow_timer)
        S.status = "录制中：%s" % "、".join(JOINT_LABELS[i] for i in joints)
        return {"FINISHED"}


def _collect_frames(context):
    """逐帧采样当前时间轴，返回 (times_ms, poses)。"""
    scene = context.scene
    arm = _rig(context)
    fps = scene.render.fps / scene.render.fps_base
    keep = scene.frame_current
    times, poses = [], []
    S.sending = True
    try:
        for f in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(f)
            times.append((f - scene.frame_start) * 1000.0 / fps)
            poses.append(rig.get_angles(arm))
    finally:
        scene.frame_set(keep)
        S.sending = False
    return times, poses


def _build_action(context):
    props = _props(context)
    times, poses = _collect_frames(context)
    window = 1  # 时间轴上的动画本身是干净的，不再去抖
    if props.interp == "ease":
        frames, _ = lulu_link.samples_to_frames(times, poses, epsilon=max(props.simplify_deg, 3.0),
                                                smooth_window=window)
    else:
        frames, _ = lulu_link.samples_to_frames(times, poses, epsilon=props.simplify_deg, smooth_window=window)
    loop = {"from": 0, "to": len(frames) - 1, "times": 1} if props.loop else None
    action = lulu_link.make_action(props.action_name, frames, interp=props.interp,
                                   emotion=None if props.emotion == "none" else props.emotion, loop=loop)
    return action


class LULU_OT_check(bpy.types.Operator):
    bl_idname = "lulu.check"
    bl_label = "检查翻倒和碰撞"

    def execute(self, context):
        scene = context.scene
        for m in [m for m in scene.timeline_markers if m.name.startswith("⚠")]:
            scene.timeline_markers.remove(m)
        times, poses = _collect_frames(context)
        bad = 0
        last = None
        for k, pose in enumerate(poses):
            problems = lulu_link.check_pose(pose)
            text = "；".join(problems)
            if problems and text != last:
                scene.timeline_markers.new("⚠ " + text, frame=scene.frame_start + k)
                bad += 1
            last = text if problems else None
        if bad:
            self.report({"WARNING"}, "发现 %d 处问题，已在时间轴标出（⚠）" % bad)
        else:
            self.report({"INFO"}, "没有发现问题")
        return {"FINISHED"}


class LULU_OT_send(bpy.types.Operator):
    bl_idname = "lulu.send"
    bl_label = "发送到真机"
    mode: EnumProperty(items=[("PREVIEW", "预览", ""), ("INSTALL", "安装", "")])

    def execute(self, context):
        if not S.link:
            self.report({"ERROR"}, "先连接噜噜")
            return {"CANCELLED"}
        action = _build_action(context)
        problems = lulu_link.validate_action(action)
        if problems:
            self.report({"ERROR"}, "；".join(problems))
            return {"CANCELLED"}
        props = _props(context)
        props.live_sync = False
        if self.mode == "PREVIEW":
            ok, msg = S.link.preview_action(action, speed=props.preview_speed)
        else:
            ok, msg = S.link.install_action(action)
        if not ok:
            self.report({"ERROR"}, "失败：%s" % msg)
            return {"CANCELLED"}
        self.report({"INFO"}, "%s成功：%s（%d 帧）" % ("预览" if self.mode == "PREVIEW" else "安装",
                                                  action["name"], len(action["frames"])))
        return {"FINISHED"}


class LULU_OT_export(bpy.types.Operator, ExportHelper):
    bl_idname = "lulu.export"
    bl_label = "导出动作 JSON"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def invoke(self, context, event):
        self.filepath = _props(context).action_name + ".json"
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        action = _build_action(context)
        problems = lulu_link.validate_action(action)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(action, f, ensure_ascii=False, indent=1)
        if problems:
            self.report({"WARNING"}, "已导出，但：" + "；".join(problems))
        else:
            self.report({"INFO"}, "已导出 %d 帧到 %s" % (len(action["frames"]), os.path.basename(self.filepath)))
        return {"FINISHED"}


class LULU_OT_import(bpy.types.Operator, ImportHelper):
    bl_idname = "lulu.import_action"
    bl_label = "导入动作 JSON"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        arm = _rig(context) or rig.build_rig(context)
        with open(self.filepath, encoding="utf-8") as f:
            action = json.load(f)
        scene = context.scene
        fps = scene.render.fps / scene.render.fps_base
        frames = lulu_link.action_frames(action)
        start = scene.frame_start
        t = 0.0
        S.sending = True
        try:
            for n, (d, angles) in enumerate(frames):
                if n:
                    t += d
                f = start + round(t * fps / 1000.0)
                rig.set_angles(arm, angles)
                rig.keyframe_angles(arm, f)
        finally:
            S.sending = False
        scene.frame_end = start + round(t * fps / 1000.0)
        props = _props(context)
        props.action_name = action.get("name", props.action_name)
        props.interp = "spline" if action.get("interp") == "spline" else "ease"
        props.loop = "loop" in action
        if action.get("emotion"):
            try:
                props.emotion = action["emotion"]
            except TypeError:
                pass
        self.report({"INFO"}, "已导入 %d 帧" % len(frames))
        return {"FINISHED"}


# ------------------------------------------------------------------ 界面

class LULU_PT_main(bpy.types.Panel):
    bl_label = "噜噜动作编辑器"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "噜噜"

    def draw(self, context):
        layout = self.layout
        props = _props(context)
        if _rig(context) is None:
            layout.operator("lulu.build_rig", icon="ARMATURE_DATA")
        else:
            layout.operator("lulu.build_rig", text="重新生成模型", icon="FILE_REFRESH")

        box = layout.box()
        box.label(text="连接", icon="LINKED")
        if not _deps_ok():
            box.operator("lulu.install_deps", icon="IMPORT")
        box.prop(props, "transport", expand=True)
        if props.transport == "USB":
            box.prop(props, "port")
        row = box.row()
        row.operator("lulu.connect", icon="PLUGIN")
        row.operator("lulu.disconnect")
        box.label(text=S.status)

        box = layout.box()
        box.label(text="实时操控", icon="ARMATURE_DATA")
        box.enabled = S.link is not None
        box.prop(props, "live_sync")
        box.prop(props, "follow")
        col = box.column(heading="上力（取消勾选可手掰）")
        for i in range(5):
            col.prop(props, "torque", index=i, text=JOINT_LABELS[i])
        box.operator("lulu.record", text="停止录制" if S.recording else "开始录制卸力关节",
                     icon="PAUSE" if S.recording else "REC", depress=S.recording)

        box = layout.box()
        box.label(text="导出", icon="EXPORT")
        box.prop(props, "action_name")
        box.prop(props, "emotion")
        box.prop(props, "interp")
        box.prop(props, "loop")
        box.prop(props, "simplify_deg")
        box.prop(props, "smooth_window")
        box.operator("lulu.check", icon="ERROR")
        row = box.row()
        row.enabled = S.link is not None
        row.operator("lulu.send", text="预览到真机", icon="PLAY").mode = "PREVIEW"
        row.operator("lulu.send", text="安装到真机", icon="CHECKMARK").mode = "INSTALL"
        box.prop(props, "preview_speed")
        row = box.row()
        row.operator("lulu.export", icon="FILE")
        row.operator("lulu.import_action", icon="FILEBROWSER")


CLASSES = (LuluProps, LULU_OT_install_deps, LULU_OT_build_rig, LULU_OT_connect, LULU_OT_disconnect,
           LULU_OT_record, LULU_OT_check, LULU_OT_send, LULU_OT_export, LULU_OT_import, LULU_PT_main)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.lulu = bpy.props.PointerProperty(type=LuluProps)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph)
    bpy.app.handlers.frame_change_post.append(_on_frame)
    _modules_dir()


def unregister():
    if S.link:
        S.link.close()
        S.link = None
    for h, fn in ((bpy.app.handlers.depsgraph_update_post, _on_depsgraph),
                  (bpy.app.handlers.frame_change_post, _on_frame)):
        if fn in h:
            h.remove(fn)
    del bpy.types.Scene.lulu
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
