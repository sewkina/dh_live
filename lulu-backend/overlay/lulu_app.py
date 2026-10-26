"""噜噜的启动入口：先装运行时补丁，再照常运行上游的 app.py。

拍照硬开关：大模型想调用拍照工具时，检查孩子/主人最近一句话里有没有明确的拍照指令
（中文、英文、高棉语）。没有就直接拒绝，不靠提示词约束。
"""

import re

PHOTO_TOOL_KEYWORD = "take_photo"

# 明确要求“看/拍”的说法。识别结果可能有同音错字，所以只列常见、不易误触的短语
PHOTO_TRIGGERS = re.compile(
    "|".join(
        [
            # 中文
            r"拍照", r"拍张", r"拍一张", r"拍个照", r"拍下", r"照相",
            r"看看这", r"看一下这", r"看下这", r"看看我", r"你看这", r"你看我",
            r"这是什么", r"我拿的是什么", r"手里是什么",
            # 英文
            r"take a (photo|picture)", r"take photo", r"look at (this|me|that)",
            r"what is this", r"what's this", r"what am i holding", r"can you see",
            # 高棉语：ថតរូប 拍照 / មើល 看 / នេះជាអ្វី 这是什么
            r"ថតរូប", r"ថត", r"មើល", r"នេះជាអ្វី", r"នេះអ្វី",
        ]
    ),
    re.IGNORECASE,
)

REFUSAL = (
    "拍照被拒绝：对方这句话没有明确要求你看东西或拍照。"
    "不要再调用拍照工具，用对方的语言正常回答；如果需要看东西，请对方明确说“看看这是什么”。"
)


def _last_user_text(conn):
    dialogue = getattr(getattr(conn, "dialogue", None), "dialogue", None) or []
    for message in reversed(dialogue):
        if getattr(message, "role", None) == "user" and not getattr(message, "is_temporary", False):
            return message.content or ""
    return ""


def _is_photo_call(function_call_data):
    calls = function_call_data.get("function_calls") or [function_call_data]
    return any(PHOTO_TOOL_KEYWORD in str(call.get("name", "")) for call in calls)


def _install_photo_gate():
    from core.providers.tools import unified_tool_handler
    from plugins_func.register import Action, ActionResponse

    handler_cls = unified_tool_handler.UnifiedToolHandler
    original = handler_cls.handle_llm_function_call
    if getattr(original, "_lulu_gated", False):
        return

    async def gated(self, conn, function_call_data):
        if _is_photo_call(function_call_data):
            text = _last_user_text(conn)
            if not PHOTO_TRIGGERS.search(text):
                self.logger.warning(f"拦截拍照：用户原话没有拍照指令: {text!r}")
                return ActionResponse(action=Action.REQLLM, result=REFUSAL, response=REFUSAL)
            self.logger.info(f"允许拍照：{text!r}")
        return await original(self, conn, function_call_data)

    gated._lulu_gated = True
    handler_cls.handle_llm_function_call = gated


if __name__ == "__main__":
    import runpy

    # 装不上就直接报错退出：宁可后端起不来，也不能让大模型随便拍照
    _install_photo_gate()
    runpy.run_path("app.py", run_name="__main__")
