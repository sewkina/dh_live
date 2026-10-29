"""在 Blender 里生成噜噜的替身模型和 5 关节骨架，以及骨架姿态和关节角度的互相换算。

骨架约定（Cascadeur / Houdini 导入的骨架只要符合这几条，也能直接导出）：
- 骨骼名 body、waist、leg_LF、leg_RF、leg_LB、leg_RB
- 静止姿态：腿垂直向下，waist 指向正前方（+Y），x 向右，z 向上
- 前腿是 waist 的子骨骼，后腿是 body 的子骨骼
"""

import math

import bpy
from mathutils import Vector

from .lulu_link import ANGLE_MAX, ANGLE_MIN, DIMS, JOINTS, STAND_ANGLES

RIG_NAME = "LuluRig"
HIP_Z = 0.0364         # 髋关节高度：站立（腿 36 度）时脚正好踩在 z=0
FRONT_LEGS = ("leg_LF", "leg_RF")


def _hip(name):
    side = -1.0 if name in ("leg_LF", "leg_LB") else 1.0
    front = name in FRONT_LEGS
    return Vector((side * DIMS["hip_dx"], DIMS["hip_dy"] if front else -DIMS["hip_dy"], HIP_Z))


def _material(name, rgba):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    if hasattr(mat, "use_nodes"):
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.node_tree else None
        if bsdf:
            bsdf.inputs["Base Color"].default_value = rgba
    return mat


def _box(name, size, location, mat, bevel=0.004):
    mesh = bpy.data.meshes.new(name)
    sx, sy, sz = (s / 2 for s in size)
    verts = [(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    return obj


def _cylinder(name, radius, depth, location, rotation, mat, verts=32):
    mesh = bpy.data.meshes.new(name)
    coords, faces = [], []
    for k in range(verts):
        a = 2 * math.pi * k / verts
        coords += [(radius * math.cos(a), radius * math.sin(a), -depth / 2),
                   (radius * math.cos(a), radius * math.sin(a), depth / 2)]
        n = (k + 1) % verts
        faces.append((2 * k, 2 * n, 2 * n + 1, 2 * k + 1))
    faces.append(tuple(2 * k for k in reversed(range(verts))))
    faces.append(tuple(2 * k + 1 for k in range(verts)))
    mesh.from_pydata(coords, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.rotation_euler = rotation
    obj.data.materials.append(mat)
    return obj


def _sphere(name, radius, location, scale, mat):
    mesh = bpy.data.meshes.new(name)
    rings, segs = 12, 24
    coords = [(0, 0, -radius)]
    for r in range(1, rings):
        phi = math.pi * r / rings - math.pi / 2
        for s in range(segs):
            th = 2 * math.pi * s / segs
            coords.append((radius * math.cos(phi) * math.cos(th), radius * math.cos(phi) * math.sin(th),
                           radius * math.sin(phi)))
    coords.append((0, 0, radius))
    faces = []
    for s in range(segs):
        faces.append((0, 1 + (s + 1) % segs, 1 + s))
    for r in range(rings - 2):
        for s in range(segs):
            a = 1 + r * segs + s
            b = 1 + r * segs + (s + 1) % segs
            faces.append((a, b, b + segs, a + segs))
    top = len(coords) - 1
    base = 1 + (rings - 2) * segs
    for s in range(segs):
        faces.append((base + s, base + (s + 1) % segs, top))
    mesh.from_pydata(coords, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    obj.scale = scale
    obj.data.materials.append(mat)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def _parent_to_bone(obj, arm, bone):
    mw = obj.matrix_world.copy()
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone
    obj.matrix_world = mw


def build_rig(context):
    """生成替身模型 + 骨架，返回骨架对象。已存在则先删掉重建。"""
    old = bpy.data.objects.get(RIG_NAME)
    if old:
        for child in list(old.children):
            bpy.data.objects.remove(child, do_unlink=True)
        bpy.data.objects.remove(old, do_unlink=True)

    coll = bpy.data.collections.get("Lulu") or bpy.data.collections.new("Lulu")
    if coll.name not in context.scene.collection.children:
        context.scene.collection.children.link(coll)

    arm_data = bpy.data.armatures.new(RIG_NAME)
    arm = bpy.data.objects.new(RIG_NAME, arm_data)
    arm["lulu_rig"] = 1
    coll.objects.link(arm)
    arm.show_in_front = True

    view_layer = context.view_layer
    view_layer.objects.active = arm
    for o in context.selected_objects:
        o.select_set(False)
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm_data.edit_bones
    body = eb.new("body")
    body.head, body.tail = Vector((0, -0.030, HIP_Z)), Vector((0, 0.0, HIP_Z))
    waist = eb.new("waist")
    waist.head, waist.tail = Vector((0, 0.0, HIP_Z)), Vector((0, 0.030, HIP_Z))
    waist.parent = body
    for name in JOINTS[:4]:
        b = eb.new(name)
        hip = _hip(name)
        b.head, b.tail = hip, hip - Vector((0, 0, DIMS["leg_len"]))
        # 让骨骼本地 X 轴对齐世界 +X，绕 X 正转时脚向前
        b.align_roll(Vector((0, 1, 0)))
        b.parent = waist if name in FRONT_LEGS else body
    bpy.ops.object.mode_set(mode="POSE")

    for name in JOINTS:
        pb = arm.pose.bones[name]
        pb.rotation_mode = "XYZ"
        idx = JOINTS.index(name)
        con = pb.constraints.new("LIMIT_ROTATION")
        con.owner_space = "LOCAL"
        if name == "waist":
            pb.lock_rotation = (True, True, False)
            con.use_limit_z = True
            con.min_z, con.max_z = math.radians(ANGLE_MIN[idx]), math.radians(ANGLE_MAX[idx])
        else:
            pb.lock_rotation = (False, True, True)
            con.use_limit_x = True
            lo, hi = math.radians(ANGLE_MIN[idx]), math.radians(ANGLE_MAX[idx])
            if name in FRONT_LEGS:
                con.min_x, con.max_x = lo, hi
            else:
                con.min_x, con.max_x = -hi, -lo
    pb_body = arm.pose.bones["body"]
    pb_body.lock_location = (True, True, True)
    pb_body.lock_rotation = (True, True, True)
    bpy.ops.object.mode_set(mode="OBJECT")

    # ---- 替身外形（按估算尺寸，换精确模型时保持骨骼名和静止姿态不变即可）
    shell = _material("Lulu_Shell", (0.93, 0.91, 0.88, 1))
    dark = _material("Lulu_Dark", (0.15, 0.15, 0.17, 1))
    screen = _material("Lulu_Screen", (0.2, 0.55, 0.9, 1))
    parts = [
        (_box("Lulu_BodyRear", (0.056, 0.046, 0.034), (0, -0.024, HIP_Z + 0.004), shell), "body"),
        (_box("Lulu_BodyFront", (0.056, 0.046, 0.034), (0, 0.024, HIP_Z + 0.004), shell), "waist"),
        (_sphere("Lulu_Head", 0.026, (0, 0.040, HIP_Z + 0.040), (1.0, 0.85, 0.95), shell), "waist"),
        (_cylinder("Lulu_Screen", 0.0139, 0.002, (0, 0.062, HIP_Z + 0.042), (math.radians(90), 0, 0), screen), "waist"),
        (_cylinder("Lulu_Tail", 0.004, 0.022, (0, -0.052, HIP_Z + 0.014), (math.radians(-60), 0, 0), dark), "body"),
    ]
    for name in JOINTS[:4]:
        hip = _hip(name)
        leg = _box("Lulu_" + name, (DIMS["leg_thick"], 0.007, DIMS["leg_len"] + 0.004),
                   hip - Vector((0, 0, DIMS["leg_len"] / 2)), dark, bevel=0.003)
        parts.append((leg, name))
    for obj, bone in parts:
        coll.objects.link(obj)
        _parent_to_bone(obj, arm, bone)

    scene = context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    if scene.render.fps not in (25, 30, 50, 60):
        scene.render.fps = 30
    set_angles(arm, STAND_ANGLES)
    view_layer.objects.active = arm
    return arm


def find_rig(context):
    obj = context.active_object
    if obj and obj.type == "ARMATURE" and all(n in obj.pose.bones for n in JOINTS):
        return obj
    for obj in context.scene.objects:
        if obj.type == "ARMATURE" and all(n in obj.pose.bones for n in JOINTS):
            return obj
    return None


def set_angles(arm, angles, joints=None):
    """把关节角度写进骨骼旋转（只适用于本插件生成的骨架）。joints 为要写的关节下标，默认全部。"""
    for i in (joints if joints is not None else range(5)):
        pb = arm.pose.bones[JOINTS[i]]
        a = math.radians(angles[i])
        if i == 4:
            pb.rotation_euler = (0.0, 0.0, a)
        elif JOINTS[i] in FRONT_LEGS:
            pb.rotation_euler = (a, 0.0, 0.0)
        else:
            pb.rotation_euler = (-a, 0.0, 0.0)


def get_angles(arm):
    """从已求值的骨骼姿态读出 5 个关节角度（度）。只看骨骼方向，所以导入的其他骨架也能用。"""
    pbs = arm.pose.bones
    body = pbs["body"].matrix
    waist = pbs["waist"].matrix
    angles = []
    for name in JOINTS[:4]:
        parent = waist if name in FRONT_LEGS else body
        d = (parent.to_3x3().inverted() @ pbs[name].matrix.to_3x3().col[1]).normalized()
        if name in FRONT_LEGS:
            angles.append(math.degrees(math.atan2(d.y, -d.z)))
        else:
            angles.append(math.degrees(math.atan2(-d.y, -d.z)))
    w = (body.to_3x3().inverted() @ waist.to_3x3().col[1]).normalized()
    angles.append(math.degrees(math.atan2(-w.x, w.y)))
    return angles


def keyframe_angles(arm, frame, joints=None):
    for i in (joints if joints is not None else range(5)):
        pb = arm.pose.bones[JOINTS[i]]
        pb.keyframe_insert("rotation_euler", index=2 if i == 4 else 0, frame=frame, group=JOINTS[i])


def delete_keys(arm, joints, frame_start, frame_end):
    for i in joints:
        pb = arm.pose.bones[JOINTS[i]]
        for f in range(int(frame_start), int(frame_end) + 1):
            try:
                pb.keyframe_delete("rotation_euler", index=2 if i == 4 else 0, frame=f)
            except RuntimeError:
                pass

