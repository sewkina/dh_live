# 噜噜工坊 · 资源库

给噜噜挑动作、表情、音色的网站：浏览、试看试听、下载，也能上传自己做的给别人下载。
同一个网页在电脑浏览器、手机浏览器和 Telegram 小程序里都能打开。

第一版只做“下载资源库”，动作编辑器、表情生成器、音色合成以后再加。

## 里面有什么

| 分类 | 现成内容 | 预览 | 下载 |
|---|---|---|---|
| 动作 | 11 个（挥手 + 10 个呆萌动作） | 网页里播放侧视动画（含腰部俯视小图） | 动作 JSON；电脑 Chrome 插 USB 可一键装进噜噜 |
| 表情 | 黑底、白底大眼表情各 18 个（宣传片和 3D 用黑底，固件里现在刷的是白底），带已选定的萌宠音效 | 圆屏动图，点“听音效” | `.eaf`、音效 `.ogg`；按标签筛选后打包，附 `emote_list.json` |
| 音色 | Edge 高棉语女声、VoxCPM 高棉语微调（默认声 / 噜噜参考音）、VoxCPM 中文噜噜音色 | 试听 | 试听音频 + 后端 TTS 配置片段 |

现成内容在 `seed/`，上传的内容和下载次数在 `data/`（不进 Git）。删除上传内容时先挪进 `data/trash/`，不直接删。

## 在电脑上启动（Windows）

双击 `start_lulu_studio.bat`。第一次会建 `.venv` 并安装依赖，然后启动在 **8020** 端口（8010 已被后端占用）。

- 本机：<http://localhost:8020>
- 同一个 WiFi/热点的手机：`http://电脑IP:8020`（热点下是 `http://172.20.10.3:8020`）。打不开时要在 Windows 防火墙放行 8020

其他系统：

```bash
pip install -r requirements.txt
python app.py
```

环境变量：

| 变量 | 作用 | 默认 |
|---|---|---|
| `LULU_STUDIO_PORT` | 端口 | 8020 |
| `LULU_STUDIO_DATA` | 上传内容、下载次数存放目录 | `./data` |
| `LULU_STUDIO_TOKEN` | 上传/删除口令；网页右上角“设置”里填 | 不设则不需要口令 |

## 一键装动作进噜噜

1. 电脑用 **Chrome 或 Edge** 打开网站（Telegram 和 Safari 不支持 USB）
2. 噜噜用**数据线**插电脑（很多线只能充电），点右上角“连接噜噜”，选噜噜的串口（CH343）
3. 打开一个动作，点“装进噜噜（USB）”：只保存，噜噜不动。“在噜噜上试播”会让噜噜马上动，要先放在平地上

用的是固件里关键帧播放器的 USB 协议（`lulu_tools/firmware` 补丁里的 `kf_link.h`），连上会显示固件版本 `lulu-kf/1`。
如果用 Blender 插件连着同一个串口，要先在 Blender 里断开。

## 当 Telegram 小程序用

Telegram 小程序要一个公网 **https** 地址。电脑上没有公网 IP，可以用 Cloudflare 免费隧道：

1. **先设上传口令**，不然拿到链接的人都能上传、删除：启动前 `set LULU_STUDIO_TOKEN=自己想一个`
2. 装 `cloudflared`，运行 `cloudflared tunnel --url http://localhost:8020`，它会给一个 `https://xxxx.trycloudflare.com` 地址（每次重启会变；要固定地址需要在 Cloudflare 绑自己的域名）
3. 在 Telegram 找 @BotFather → 选噜噜的机器人 → Bot Settings → Menu Button（或 Configure Mini App）→ 填上面的 https 地址

在 Telegram 里能浏览、试听、下载，主题跟随 Telegram，左上角返回键能关掉详情。Telegram 里不能连 USB，装动作要用电脑 Chrome。
以后上了云服务器，直接用服务器域名，不用隧道。

## 上传格式

- **动作**：`.json`，和固件关键帧播放器同一种格式（见 `lulu_tools/README.md`）。最多 40 帧、小于 4000 字节、`p` 是 5 个 -3000~3000 的整数、`d` 是 20~10000 毫秒，`name` 只能用英文字母、数字、下划线。Blender 插件导出的文件可以直接传
- **表情**：`.eaf`（240×240，噜噜屏幕格式），会检查文件头和校验和；不传预览图会自动从 `.eaf` 生成 GIF。可附一个音效
- **音色**：一段试听音频（mp3/ogg/wav/m4a/opus，10 MB 以内），可附一个 JSON 配置

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/items?type=actions\|emotes\|voices&q=` | 列表 |
| GET | `/api/items/{type}/{id}` | 单个 |
| GET | `/files/{type}/{id}/{文件名}?dl=1` | 文件（`dl=1` 作为下载并计数） |
| GET | `/api/items/{type}/{id}/zip` | 单个打包 |
| GET | `/api/pack/{type}.zip?ids=a,b` | 多个打包，不填 `ids` 就是整类 |
| POST | `/api/upload` | 上传（表单），设了口令要带 `X-Lulu-Token` 头 |
| DELETE | `/api/items/{type}/{id}` | 删除上传的内容（官方内容不能删） |

## 测试

```bash
pip install pytest httpx
python -m pytest -q tests
```

## 以后要加

- 动作编辑器（网页 3D + 手掰录制）、表情生成器、音色合成和“设为噜噜音色”
- Telegram 里经后端把动作推给噜噜（现在后端没有对外的推送接口）
- 上了云服务器后：账号、上传审核、举报
