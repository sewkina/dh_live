"""生成一组呆萌动作（关键帧 JSON），并用 lulu_link 的翻倒/碰撞检查逐帧过一遍。

运行：python make_cute_actions.py [输出目录]
角度约定：腿 0 度垂直向下、90 度水平，站立 36 度；前腿正值脚往前，后腿正值脚往后；腰正值上半身向左转。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "blender", "lulu_animator"))
import lulu_link as L  # noqa: E402

S = 36  # 站立


def pose(lf=S, rf=S, lb=S, rb=S, w=0):
    return [lf, rf, lb, rb, w]


STAND = pose()

ACTIONS = {
    # 歪头卖萌：上半身慢慢往左扭、停一下，再往右扭、停一下，回正
    "cute_tilt": dict(emotion="cute", frames=[
        (500, pose(lf=44, rf=44, w=38)), (500, pose(lf=44, rf=44, w=38)),
        (700, pose(lf=44, rf=44, w=-38)), (500, pose(lf=44, rf=44, w=-38)),
        (500, STAND),
    ]),
    # 伸懒腰：前腿往前伸、胸口压低屁股翘高（像小狗邀玩），停住，再换后腿往后蹬
    "stretch": dict(emotion="sleepy", frames=[
        (900, pose(lf=88, rf=88, lb=12, rb=12)), (1200, pose(lf=88, rf=88, lb=12, rb=12)),
        (800, pose(lf=22, rf=22, lb=80, rb=80)), (900, pose(lf=22, rf=22, lb=80, rb=80)),
        (700, STAND),
    ]),
    # 摇屁股：腰快速左右摇，中间三下循环
    "wiggle": dict(emotion="happy", frames=[
        (250, pose(w=30)), (200, pose(w=-30)), (200, pose(w=30)), (200, pose(w=-30)),
        (300, STAND),
    ], loop={"from": 1, "to": 2, "times": 3}),
    # 开心跺脚：两条前腿轮流往前抬一下，像小碎步
    "happy_tap": dict(emotion="celebrate", frames=[
        (220, pose(lf=95, w=8)), (200, pose(w=0)),
        (200, pose(rf=95, w=-8)), (200, pose(w=0)),
        (300, STAND),
    ], loop={"from": 0, "to": 3, "times": 3}),
    # 害羞扭扭：低头往左扭，小幅扭两下，再慢慢抬头
    "shy_twist": dict(emotion="shy", frames=[
        (600, pose(lf=72, rf=72, w=38)), (300, pose(lf=72, rf=72, w=22)),
        (300, pose(lf=72, rf=72, w=38)), (300, pose(lf=72, rf=72, w=22)),
        (800, pose(lf=72, rf=72, w=38)), (700, STAND),
    ]),
    # 趴下撒娇：慢慢趴下，趴着左右蹭两下，再站起来
    "lie_cuddle": dict(emotion="loving", frames=[
        (1000, pose(85, 85, 85, 85)), (500, pose(85, 85, 85, 85, 32)),
        (500, pose(85, 85, 85, 85, -32)), (500, pose(85, 85, 85, 85, 32)),
        (500, pose(85, 85, 85, 85)), (900, STAND),
    ]),
    # 好奇探头：前腿往后收、身体前倾探头看，左右张望，再缩回来
    "curious_peek": dict(emotion="curious", frames=[
        (600, pose(lf=5, rf=5, lb=62, rb=62)), (600, pose(lf=5, rf=5, lb=62, rb=62, w=32)),
        (700, pose(lf=5, rf=5, lb=62, rb=62, w=-32)), (400, pose(lf=5, rf=5, lb=62, rb=62)),
        (700, STAND),
    ]),
}


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ok = True
    for name, spec in ACTIONS.items():
        action = L.make_action(name, spec["frames"], emotion=spec["emotion"])
        if "loop" in spec:
            action["loop"] = spec["loop"]
        probs = L.validate_action(action)
        for k, (_, angles) in enumerate(spec["frames"]):
            probs += ["第 %d 帧：%s" % (k, p) for p in L.check_pose(angles)]
        total = sum(d for d, _ in spec["frames"])
        print("%-13s %-9s %2d 帧 %.1f 秒 %s" % (name, spec["emotion"], len(spec["frames"]), total / 1000,
                                             "OK" if not probs else "；".join(probs)))
        ok &= not probs
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(action, f, ensure_ascii=False, indent=1)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))) else 1)
