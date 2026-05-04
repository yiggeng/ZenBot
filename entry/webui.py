import asyncio
import os
import sys
import json
import threading
from datetime import datetime

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

# 确保项目根目录在 path 里
ENTRY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ENTRY_DIR)
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from zenbot.core.config import DB_PATH
from zenbot.core.multi_agent import create_multi_agent_app

# ─────────────── 全局状态 ───────────────

_app = None          # LangGraph app（懒加载，首次对话时初始化）
_memory_ctx = None   # AsyncSqliteSaver 上下文管理器
_memory = None       # checkpointer 实例
_loop = None         # 后台事件循环
_stop_event = threading.Event()  # 用于中断当前对话轮次


def _get_loop():
    """获取或创建后台事件循环（在独立线程里跑，避免和 Gradio 的循环冲突）"""
    global _loop
    if _loop is None or not _loop.is_running():
        import threading
        _loop = asyncio.new_event_loop()
        t = threading.Thread(target=_loop.run_forever, daemon=True)
        t.start()
    return _loop


async def _ensure_app():
    """懒加载：首次调用时初始化 LangGraph app 和 SQLite checkpointer"""
    global _app, _memory_ctx, _memory
    if _app is not None:
        return
    provider = os.getenv("DEFAULT_PROVIDER", "aliyun")
    model = os.getenv("DEFAULT_MODEL", "qwen-plus")
    _memory_ctx = AsyncSqliteSaver.from_conn_string(DB_PATH)
    _memory = await _memory_ctx.__aenter__()
    _app = create_multi_agent_app(
        provider_name=provider,
        model_name=model,
        checkpointer=_memory
    )


# ─────────────── 核心对话逻辑 ───────────────

async def _load_chat_history(thread_id: str) -> list:
    """从 LangGraph checkpointer 读取指定 thread_id 的聊天记录，返回 Gradio chatbox 格式"""
    await _ensure_app()
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = await _memory.aget_tuple(config)
    if checkpoint_tuple is None:
        return []
    messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage) and msg.content:
            history.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content:
            history.append({"role": "assistant", "content": str(msg.content)})
    return history

async def _run_turn(user_input: str, thread_id: str):
    """
    驱动一轮对话，yield (role, content) 流式输出。
    role: "assistant" | "approval" | "log"
    """
    await _ensure_app()
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "user_input": user_input,
        "messages": [],
        "summary": "",
        "final_answer": "",
    }

    pending_resume = None   # 存放 interrupt 后用户的回复

    while True:
        got_interrupt = False
        interrupt_val = {}

        async for event in _app.astream(
            pending_resume if pending_resume is not None else inputs,
            config,
            stream_mode="updates"
        ):
            for node_name, node_data in event.items():
                if node_name == "__interrupt__":
                    got_interrupt = True
                    item = node_data[0] if node_data else None
                    interrupt_val = getattr(item, "value", None) or {}
                    if not isinstance(interrupt_val, dict):
                        interrupt_val = {}
                else:
                    # 把节点进度通知给 UI
                    text = _format_node_event(node_name, node_data)
                    if text:
                        yield ("log", text)

                    # 提取最终回复
                    if node_name == "multi_subgraph":
                        answer = (node_data or {}).get("final_answer", "")
                        if answer and answer != "__replan__":
                            yield ("assistant", answer)

        if got_interrupt:
            # 把审批方案发给 UI，等待用户回复
            plan_text = interrupt_val.get("plan", "")
            task_count = interrupt_val.get("task_count", 0)
            mode_label = interrupt_val.get("mode", "并行")
            itype = interrupt_val.get("type", "plan_approval")

            if itype == "plan_approval":
                label = "重新拆解出" if interrupt_val.get("replan") else "拆解出"
                approval_msg = (
                    f"**✦ Planner {label} {task_count} 个{mode_label}子任务：**\n\n"
                    f"```\n{plan_text}\n```\n\n"
                    f"确认执行？请输入 **y** 确认或 **n** 拒绝"
                )
            else:
                approval_msg = interrupt_val.get("prompt", "请输入改进建议：")

            yield ("approval", approval_msg)
            return   # 暂停，等 UI 调 resume_turn()
        else:
            break   # 正常结束


async def _resume_turn(user_reply: str, thread_id: str):
    """interrupt 之后用户回复，Resume 执行"""
    await _ensure_app()
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        got_interrupt = False
        interrupt_val = {}

        async for event in _app.astream(
            Command(resume=user_reply),
            config,
            stream_mode="updates"
        ):
            for node_name, node_data in event.items():
                if node_name == "__interrupt__":
                    got_interrupt = True
                    item = node_data[0] if node_data else None
                    interrupt_val = getattr(item, "value", None) or {}
                    if not isinstance(interrupt_val, dict):
                        interrupt_val = {}
                else:
                    text = _format_node_event(node_name, node_data)
                    if text:
                        yield ("log", text)

                    if node_name == "multi_subgraph":
                        answer = (node_data or {}).get("final_answer", "")
                        if answer and answer != "__replan__":
                            yield ("assistant", answer)

        if got_interrupt:
            plan_text = interrupt_val.get("plan", "")
            task_count = interrupt_val.get("task_count", 0)
            mode_label = interrupt_val.get("mode", "并行")
            itype = interrupt_val.get("type", "plan_approval")
            if itype == "plan_approval":
                label = "重新拆解出" if interrupt_val.get("replan") else "拆解出"
                approval_msg = (
                    f"**✦ Planner {label} {task_count} 个{mode_label}子任务：**\n\n"
                    f"```\n{plan_text}\n```\n\n"
                    f"确认执行？请输入 **y** 确认或 **n** 拒绝"
                )
            else:
                approval_msg = interrupt_val.get("prompt", "请输入改进建议：")
            yield ("approval", approval_msg)
            return
        else:
            break


def _format_node_event(node_name: str, node_data: dict) -> str:
    """把节点事件格式化成 monitor 日志文本"""
    if not node_data:
        return ""
    ts = datetime.now().strftime("%H:%M:%S")

    if node_name == "planner":
        tasks = (node_data or {}).get("tasks", [])
        if tasks:
            lines = "\n".join([f"  #{t['id']} {t['desc'][:50]}" for t in tasks])
            return f"`{ts}` 📋 Planner 拆解了 {len(tasks)} 个子任务\n```\n{lines}\n```"

    elif node_name == "worker":
        return f"`{ts}` ⚙️ Worker 执行中..."

    elif node_name == "multi_subgraph":
        answer = (node_data or {}).get("final_answer", "")
        if answer and answer != "__replan__":
            return f"`{ts}` ✅ Multi-Agent 汇总完成"

    return ""


# ─────────────── 读取实时日志 ───────────────

def _read_recent_logs(thread_id: str = "zenbot_main", n: int = 30) -> str:
    """读取最近 n 条审计日志，格式化为 Markdown"""
    log_path = os.path.join(PROJECT_ROOT, "logs", f"{thread_id}.jsonl")
    if not os.path.exists(log_path):
        return "*暂无日志*"

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    recent = lines[-n:] if len(lines) > n else lines
    out = []
    for line in reversed(recent):
        try:
            d = json.loads(line.strip())
            ts_str = d.get("ts", "")
            try:
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1] + "+00:00"
                ts = datetime.fromisoformat(ts_str).astimezone().strftime("%H:%M:%S")
            except Exception:
                ts = ts_str[-8:]

            event = d.get("event", "")
            content = d.get("content", "")

            if event == "tool_call":
                tool = d.get("tool", "")
                out.append(f"`{ts}` 🔧 **{tool}**")
            elif event == "tool_result":
                tool = d.get("tool", "")
                summary = d.get("result_summary", "")[:80]
                out.append(f"`{ts}` 📦 `{tool}` → {summary}")
            elif event == "ai_message":
                preview = content[:100].replace("\n", " ")
                out.append(f"`{ts}` 💬 {preview}")
            elif event == "system_action":
                out.append(f"`{ts}` ⚙️ {content[:80]}")
        except Exception:
            continue

    return "\n\n".join(out) if out else "*暂无日志*"


# ─────────────── Gradio UI ───────────────

def _run_async(coro):
    """在后台事件循环里同步跑一个协程，返回结果"""
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=300)


def _collect_async_gen(async_gen):
    """把异步生成器的所有输出收集成列表，支持 _stop_event 中断"""
    loop = _get_loop()

    async def _collect():
        results = []
        async for item in async_gen:
            if _stop_event.is_set():
                break
            results.append(item)
        return results

    future = asyncio.run_coroutine_threadsafe(_collect(), loop)
    return future.result(timeout=300)


def build_ui():
    with gr.Blocks(
        title="ZenBot",
    ) as demo:

        # ── 标题 ──
        gr.Markdown(
            """
            # ⚡ ZenBot
            **聪明、高效、说话自然的 AI 助手** · 自动意图路由 · 多智能体并行执行
            """,
        )

        # ── 会话状态 ──
        thread_id_state = gr.State("zenbot_main")
        awaiting_approval = gr.State(False)   # 是否正在等待 planner 审批

        with gr.Row():
            # ── 左侧：对话 ──
            with gr.Column(scale=3):
                chatbox = gr.Chatbot(
                    label="对话",
                    elem_id="chatbox",
                    height=600,
                    avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=zenbot"),
                    buttons=["copy_all"],
                    render_markdown=True,
                    layout="bubble",
                    placeholder="输入任何问题，ZenBot 自动判断是直接回答还是启动多智能体执行...",
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="输入消息，Enter 发送...",
                        show_label=False,
                        scale=5,
                        submit_btn=True,
                        lines=1,
                    )
                with gr.Row():
                    new_session_btn = gr.Button("🆕 新会话", size="sm", variant="secondary")
                    stop_btn = gr.Button("⏹ 停止", size="sm", variant="stop")
                    clear_btn = gr.Button("🗑️ 清空显示", size="sm", variant="secondary")
                    session_label = gr.Markdown("`会话：zenbot_main`")

            # ── 右侧：Monitor ──
            with gr.Column(scale=2):
                gr.Markdown("### 📡 实时监控")
                session_selector = gr.Dropdown(
                    label="切换历史会话",
                    choices=["zenbot_main"],
                    value="zenbot_main",
                    interactive=True,
                    allow_custom_value=False,
                )
                monitor_box = gr.Markdown(
                    value="*等待对话...*",
                    elem_id="monitor",
                )
                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新日志", size="sm", variant="secondary")
                    delete_session_btn = gr.Button("🗑️ 删除会话", size="sm", variant="stop")

        # ─────────────── 事件处理 ───────────────

        def handle_send(message: str, history: list, thread_id: str, is_approval: bool):
            """处理用户发送消息"""
            if not message.strip():
                yield history, "", thread_id, is_approval, _read_recent_logs(thread_id)
                return

            _stop_event.clear()

            # 用户气泡
            history = history + [{"role": "user", "content": message}]
            yield history, "", thread_id, is_approval, _read_recent_logs(thread_id)

            # 选择走正常轮次还是 resume
            if is_approval:
                events = _collect_async_gen(_resume_turn(message, thread_id))
            else:
                events = _collect_async_gen(_run_turn(message, thread_id))

            new_approval = False
            for role, content in events:
                if role == "assistant":
                    history = history + [{"role": "assistant", "content": content}]
                    yield history, "", thread_id, new_approval, _read_recent_logs(thread_id)
                elif role == "approval":
                    # Planner 审批提示，用特殊样式显示
                    history = history + [{"role": "assistant", "content": content}]
                    new_approval = True
                    yield history, "", thread_id, new_approval, _read_recent_logs(thread_id)
                elif role == "log":
                    # 进度提示，追加到最后一条 assistant 消息，或新建
                    if history and history[-1]["role"] == "assistant":
                        history[-1]["content"] += f"\n\n<div class='system-msg'>{content}</div>"
                    else:
                        history = history + [{"role": "assistant", "content": f"<div class='system-msg'>{content}</div>"}]
                    yield history, "", thread_id, new_approval, _read_recent_logs(thread_id)

            yield history, "", thread_id, new_approval, _read_recent_logs(thread_id)

        def _list_sessions() -> list:
            """扫描 logs/ 目录，返回所有 .jsonl 文件对应的 thread_id 列表（按修改时间倒序）"""
            log_dir = os.path.join(PROJECT_ROOT, "logs")
            if not os.path.exists(log_dir):
                return ["zenbot_main"]
            files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
            files.sort(key=lambda f: os.path.getmtime(os.path.join(log_dir, f)), reverse=True)
            sessions = [f[:-6] for f in files]  # 去掉 .jsonl 后缀
            return sessions if sessions else ["zenbot_main"]

        def handle_new_session(history: list):
            """开启新会话"""
            import random, string
            suffix = "".join(random.choices(string.digits, k=4))
            new_id = f"zenbot_main_{suffix}"
            sessions = _list_sessions()
            # 新会话可能尚未有日志文件，手动插到列表头部
            if new_id not in sessions:
                sessions = [new_id] + sessions
            return [], new_id, f"`会话：{new_id}`", False, gr.update(choices=sessions, value=new_id), "*等待对话...*"

        def handle_switch_session(selected_id: str, thread_id: str):
            """切换到历史会话：加载该会话的聊天记录 + Monitor 日志"""
            chat_history = _run_async(_load_chat_history(selected_id))
            return selected_id, f"`会话：{selected_id}`", False, chat_history, _read_recent_logs(selected_id)

        def handle_clear(history):
            return []

        def handle_refresh(thread_id: str):
            sessions = _list_sessions()
            return _read_recent_logs(thread_id), gr.update(choices=sessions)

        async def _delete_session_async(thread_id: str):
            """从 SQLite checkpointer 删除指定 thread_id 的所有 checkpoint"""
            await _ensure_app()
            conn = _memory.conn
            await conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            await conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = ?", (thread_id,))
            await conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = ?", (thread_id,))
            await conn.commit()

        def handle_delete_session(thread_id: str):
            """删除当前会话：清除 SQLite checkpoint + jsonl 日志，切换到下一个可用会话"""
            # 不允许删除默认会话（防止误操作删光）
            sessions = _list_sessions()

            # 删 SQLite checkpoint
            try:
                _run_async(_delete_session_async(thread_id))
            except Exception:
                pass

            # 删 jsonl 日志
            log_path = os.path.join(PROJECT_ROOT, "logs", f"{thread_id}.jsonl")
            if os.path.exists(log_path):
                os.remove(log_path)

            # 切换到下一个会话
            sessions = _list_sessions()
            next_id = sessions[0] if sessions else "zenbot_main"
            if next_id == thread_id:
                next_id = "zenbot_main"
                sessions = [next_id]

            chat_history = _run_async(_load_chat_history(next_id))
            return (
                next_id,
                f"`会话：{next_id}`",
                False,
                chat_history,
                gr.update(choices=sessions, value=next_id),
                _read_recent_logs(next_id),
            )

        def handle_stop():
            """中断当前对话轮次"""
            _stop_event.set()

        # ── 绑定事件 ──
        user_input.submit(
            fn=handle_send,
            inputs=[user_input, chatbox, thread_id_state, awaiting_approval],
            outputs=[chatbox, user_input, thread_id_state, awaiting_approval, monitor_box],
        )

        new_session_btn.click(
            fn=handle_new_session,
            inputs=[chatbox],
            outputs=[chatbox, thread_id_state, session_label, awaiting_approval, session_selector, monitor_box],
        )

        session_selector.change(
            fn=handle_switch_session,
            inputs=[session_selector, thread_id_state],
            outputs=[thread_id_state, session_label, awaiting_approval, chatbox, monitor_box],
        )

        clear_btn.click(fn=handle_clear, inputs=[chatbox], outputs=[chatbox])
        stop_btn.click(fn=handle_stop, inputs=[], outputs=[])
        refresh_btn.click(fn=handle_refresh, inputs=[thread_id_state], outputs=[monitor_box, session_selector])

        delete_session_btn.click(
            fn=handle_delete_session,
            inputs=[thread_id_state],
            outputs=[thread_id_state, session_label, awaiting_approval, chatbox, session_selector, monitor_box],
        )

        # 页面加载时，填充历史会话列表
        demo.load(
            fn=lambda: gr.update(choices=_list_sessions()),
            outputs=[session_selector],
        )

    return demo


def main():
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue="purple",
            neutral_hue="slate",
        ),
        css="""
        #chatbox { height: 600px; }
        #monitor { height: 600px; overflow-y: auto; }
        .system-msg { color: #a78bfa; font-size: 0.85em; }
        footer { display: none !important; }
        """
    )


if __name__ == "__main__":
    main()
