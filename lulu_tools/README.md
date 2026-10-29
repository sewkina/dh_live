# 噜噜动作工具

这里有两部分，配合使用：

| 目录 | 内容 |
|---|---|
| `firmware/rig_omni_keyframe.patch` | 固件补丁：关键帧播放器、按键示教录制、实时操控指令（USB + 蓝牙）、新 MCP 工具 |
| `blender/lulu_animator/` | Blender 插件：噜噜模型和骨架、实时操控真机、手掰录制、翻倒检查、导入导出动作 JSON |
| `emotes/` | 表情动画生成（第二批是 `make_emotes_v2.py`：会变形的大眼睛，黑底发光 / 白底两套） |
| `sfx/` | 表情音效：机器人风电子音 `make_sfx_robot.py`、用 ElevenLabs 生成萌宠声 `elevenlabs_sfx.py`、把自己录的声音修成萌宠声 `cute_voice.py` |

> 状态：固件只在云端做过语法检查，没有用 ESP-IDF 完整编译，也没有上真机测试。尺寸、关节方向、重心都是估算值，第一次上真机时需要核对（见文末“上真机要核对的事”）。

## 一、固件

补丁基于 [rig_omni](https://github.com/LuwuDynamics/rig_omni) 提交 `c2b42cd`，只改 `main/boards/puppy/`。

```powershell
cd D:\Xgo\rig_omni            # 你的固件源码目录
git apply --check D:\Xgo\dh_live\lulu_tools\firmware\rig_omni_keyframe.patch
git apply D:\Xgo\dh_live\lulu_tools\firmware\rig_omni_keyframe.patch
idf.py build flash monitor
```

如果你的源码已经改过 `puppy_board.cc` 等文件导致打不上，用 `git apply --3way` 再手动合并冲突。

### 新增内容

- `keyframe_player.*`：关键帧动作播放器 + 示教录制 + 实时操控
- `kf_link.*`：动作编辑器连接协议，USB 和蓝牙共用
- `robot.cc`：控制循环接入播放器；录制中三连击改为“保存”，不再进标定
- `puppy_board.cc`：新 MCP 工具、录制模式下按键单击记帧、启动 USB 连接
- `ble_remote_control.cc`：蓝牙遥控通道增加 ORDER=0x20 的编辑器命令
- `lulu_sfx.*` + `sfx_*.ogg`：换表情时播短音效（见下面“表情音效”）

### 动作 JSON 格式

```json
{
  "name": "wave2",
  "emotion": "happy",
  "interp": "spline",
  "frames": [
    {"d": 600, "p": [-100, 700, -600, 600, 200]},
    {"d": 200, "p": [-100, -400, -600, 600, 400]},
    {"d": 200, "p": [-100, -200, -600, 600, 400]}
  ],
  "loop": {"from": 1, "to": 2, "times": 3}
}
```

- `p`：5 个舵机相对零点的偏移，顺序是左前、右前、左后、右后、腰（和原厂 `set_motor_pos` 一样）。站立是 `[-600, 600, -600, 600, 0]`
- `d`：从上一帧走到这一帧的毫秒数。第一帧的 `d` 是从当前姿态过渡过去的时间
- `interp`：`"spline"` 连续不停顿（Catmull-Rom 样条）；不写就是每帧慢起慢停
- `loop`：第 from 到 to 帧重复 times 次；播放时可以用 `times` 参数覆盖
- `sound`（可选）：播放时配的音效名（如 `"woof"`、`"happy"`），`"none"` 表示不出声；不写就按 `emotion` 自动配
- 限制：最多 40 帧，JSON 小于 4000 字节，最多存 32 个动作（存在 NVS）。超出安全范围的角度会被截断

### 新 MCP 工具（后端 LLM 用）

| 工具 | 说明 |
|---|---|
| `self.dog.play(action, intensity, speed, times)` | 播放内置动作（wave、sit、hug 等 14 个）或自定义动作。幅度、速度是百分比 |
| `self.dog.list_actions()` | 列出自定义动作 |
| `self.dog.install_action(action_json)` | 安装或覆盖动作，后端可以直接推送 LLM 生成的动作 |
| `self.dog.remove_action(action)` | 删除动作 |
| `self.dog.teach(mode, name)` | 语音触发示教录制：start / frame / save / cancel |
| `self.dog.set_sound_effects(mode)` | 表情音效：0 关闭，1 只在空闲时播（默认），2 总是播 |

原来的 17 个单动作工具保留，建议后端提示词里优先用 `self.dog.play`。

### 按键示教录制（不用电脑）

1. 对噜噜说“开始录制动作，名字叫 xxx”（后端调用 `self.dog.teach(start, xxx)`），舵机卸力，屏幕显示 scanning 表情
2. 用手摆好姿势，**单击**顶部按键记一帧（会“啵”一声），再摆下一个姿势，再单击……两次单击的间隔就是这两帧之间的时间
3. 录完后停 2 秒，**三连击**保存（成功会有提示音）。三连击本身产生的点击会被自动去掉
4. 录制中长按不会触发 NVS 重置

### 表情音效

- 固件里的显示类换成 `LuluDisplay`（继承原厂 `EmoteDisplay`），每次换表情时查表播对应的 `sfx_<名字>.ogg`
- 同一个表情不重复播，两次音效至少隔 1.2 秒；平静、在听、思考不配音效
- 默认只在空闲时播。原因：后端发来的表情通常在噜噜开始说话的同时到达，这时播会盖住说话；聆听时播会被麦克风收进去
- 换音效：生成同名 `sfx_xxx.ogg` 覆盖 `main/boards/puppy/` 里的文件，重新编译。三种来源：
  - `sfx/make_sfx_robot.py`：机器人风电子音（补丁里自带的就是这套）
  - `sfx/elevenlabs_sfx.py`：在你电脑上用 ElevenLabs 账号按描述生成，每个表情出 3 版挑（先设环境变量 `ELEVENLABS_API_KEY`）
  - `sfx/cute_voice.py`：处理你自己的录音
  ```powershell
  python cute_voice.py 录音文件夹 --out out          # 文件名就是表情名，happy.m4a → sfx_happy.ogg
  python cute_voice.py 我的录音.m4a happy --pitch 1.9  # 单个文件，调升调倍数
  ```
  预设：`pet`（默认，像小动物）、`baby`（像小孩）、`tiny`（更尖更小）

## 二、Blender 插件

需要 Blender 4.2 或更新版本（在 5.2 上测过）。

### 安装

1. 把 `blender/lulu_animator` 文件夹打成 zip（或者用项目文件里现成的 `lulu_animator.zip`）
2. Blender → 编辑 → 偏好设置 → 插件 → 右上角下拉 → 从磁盘安装 → 选 zip → 勾选启用
3. 3D 视图按 N 打开侧边栏，找到“噜噜”标签
4. 第一次点“安装依赖（pyserial、bleak）”，装完重启 Blender

### 用法

- **生成噜噜模型**：生成替身模型和 5 个关节的骨架（只能绕一个轴转，带限位）。站立姿态在第 1 帧
- **连接**：选 USB（刷机那根线，COMx）或蓝牙（先对噜噜说“开启蓝牙遥控”），点“连接噜噜”。显示“已连接（lulu-kf/1）”说明固件已更新
- **实时同步到真机**：勾上后，转动骨骼或拖时间轴，真机跟着动
- **跟随真机**：取消勾选某个关节的“上力”，那个关节就卸力了，用手掰真机，Blender 里的骨骼跟着动
- **分轨录制**：把要手掰的关节取消上力，点“开始录制卸力关节”。时间轴从当前帧开始实时播放，上力的关节按已有动画动，卸力的关节记录你手掰的动作。播到结尾或再点一次就停止，自动去抖、精简成关键帧。可以先录前腿，再录后腿
- **检查翻倒和碰撞**：逐帧检查，有问题的位置在时间轴上加 ⚠ 标记
- **预览到真机 / 安装到真机**：预览只播一次不保存；安装后可以用 `self.dog.play` 调用
- **导出 / 导入动作 JSON**：和固件同一种格式

导出时会逐帧采样再精简：“连续（样条）”适合舞蹈这类连续动作，“每帧停顿（缓动）”适合点头这类一下一下的动作。“精简误差”越大帧越少。

### 角度约定

骨骼角度和 MCP 工具 `self.dog.set_motor_angle` 一致：腿 0 度垂直向下，90 度水平，站立约 36 度；前腿正值向前抬，后腿正值向后抬；腰正值上半身向左转。

### 用 Codex 换精确外形 / 从 Cascadeur、Houdini 导入

插件只认骨架，不管外形。替换外形或导入其他软件的动画时，骨架符合这几条就能直接导出：

- 骨骼名：`body`、`waist`、`leg_LF`、`leg_RF`、`leg_LB`、`leg_RB`
- 静止姿态：腿垂直向下，`waist` 指向正前方（+Y），X 向右，Z 向上，单位米
- 前腿的父骨骼是 `waist`，后腿的父骨骼是 `body`
- 外壳网格用“骨骼父级”挂到对应骨骼上

给 Codex 的任务可以这样写：“在 Blender 里按卡尺尺寸重做 Lulu_ 开头的外壳网格，挂到 LuluRig 的对应骨骼上，不改骨骼。”

### 不开 Blender 测试连接

`lulu_link.py` 不依赖 Blender（Houdini 里也能用）：

```powershell
pip install pyserial bleak
python lulu_link.py ports
python lulu_link.py ping COM3
python lulu_link.py play wave.json COM3      # 预览
python lulu_link.py install wave.json COM3   # 安装
```

## 三、连接协议

见 `firmware` 补丁里的 `kf_link.h`。

- USB：115200 波特率，一行一条，电脑发 `@L <hex>`，噜噜回 `@R <hex>`，其他行是固件日志。噜噜的 GPIO19/20 被屏幕占用，所以 USB-C 口是转串口芯片接 UART0，和日志共用
- 蓝牙：原厂遥控服务 0xFFF0（写 0xFFF2，通知 0xFFF1），帧格式 `55 00 LEN ORDER PAYLOAD CHK 00 AA`，编辑器命令 ORDER=0x20，回复 ORDER=0x21
- 实时操控 15 秒收不到指令会自动退出，恢复站立。插件每 5 秒自动发一次保活

## 上真机要核对的事

1. 编译通过（云端只做了语法检查）
2. 连接后 ping 有回应；打开串口时噜噜不应重启（如果重启，说明 DTR/RTS 接了复位电路，需要再调）
3. **关节方向**：在 Blender 里把左前腿抬到 90 度，看真机是不是左前腿向前抬平。方向反了就改 `lulu_link.py` 里的 `angle_to_offset`
4. **安全范围**：固件里 `kMinPos` / `kMaxPos` 是按角度估的，看极限位置会不会顶到外壳
5. 实时操控的延迟（USB 和蓝牙各测一次）
6. 部分关节卸力时，给其他关节发位置指令会不会让卸力的关节重新上力（EM3 舵机的行为没有文档确认）
7. 用卡尺量 `DIMS` 里的尺寸，翻倒检查才准
8. 表情音效：空闲时让它做个动作（比如按键触发），听有没有音效；说话中切表情应该没声音。模式 2 下音效和说话的先后顺序要实际听一下
9. 新表情放进 `emoji/` 后，资源分区（8 MB）是否放得下：编译日志里看 assets 大小
