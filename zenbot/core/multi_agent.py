import json
import operator
import os
from typing import Annotated, List, Literal, TypedDict

from langchain_core.runnables import RunnableConfig

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, RemoveMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Send, interrupt

from .config import MEMORY_DIR
from .context import MultiState, trim_context_messages
from .logger import audit_logger
from .provider import get_provider
from .skill_loader import load_dynamic_skills
from .tools.builtins import BUILTIN_TOOLS


# ─────────────────────────── Worker State ───────────────────────────

class WorkerState(TypedDict):
    task_id: int
    task_desc: str
    prev_results: List[str]                                          # 上一阶段的产出，由 MultiState 经 Send() 传入
    worker_messages: Annotated[List[BaseMessage], add_messages]
    worker_results: Annotated[List[str], operator.add]


# ─────────────────────────── 工厂函数 ───────────────────────────

def create_multi_agent_app(
    provider_name: str = "openai",
    model_name: str = "gpt-4o-mini",
    checkpointer=None
):
    dynamic_tools = load_dynamic_skills()
    actual_tools = BUILTIN_TOOLS + dynamic_tools
    dynamic_skill_names = [t.name for t in dynamic_tools]

    llm = get_provider(provider_name=provider_name, model_name=model_name)
    llm_with_tools = llm.bind_tools(actual_tools)
    tool_node = ToolNode(actual_tools)

    def _load_profile() -> str:
        profile_path = os.path.join(MEMORY_DIR, "user_profile.md")
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    return content
        return "暂无记录"

    def _base_sys_prompt(summary: str = "") -> str:
        profile = _load_profile()
        skills_section = ""
        if dynamic_skill_names:
            skills_list = "\n".join([f"- {n}" for n in dynamic_skill_names])
            skills_section = (
                f"\n【当前已加载的扩展技能】\n{skills_list}\n"
                f"使用技能时先 mode='help' 读说明书，再按说明书步骤执行。\n"
            )
        summary_section = f"\n【近期对话上下文】\n{summary}\n" if summary else ""
        return (
            "你是 ZenBot，一个聪明、高效、说话自然的 AI 助手。\n"
            "【重要】收到用户请求后，直接执行或回答，禁止反问、禁止要求用户澄清、禁止输出'请告诉我'等追问性文字。\n"
            "【重要】执行过程中禁止输出'我将要...'、'接下来...'、'第一步...'等计划性文字，直接调用工具，完成后再回复用户。\n"
            "需要工具时直接调用，完成后给出简洁结果。\n"
            "你运行在沙盒中，所有文件操作限制在 office 目录内。\n"
            f"{skills_section}{summary_section}"
            f"\n用户画像：{profile}"
        )

    # ─────────────── 意图路由节点 ───────────────
    def router_node(state: MultiState, config: RunnableConfig) -> dict:
        """判断是普通对话还是需要多任务并行；同时做滑动窗口压缩"""
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        # ── 每轮入口：重置单轮瞬态字段，防止上一轮 multi 残留污染 ──
        state_updates: dict = {
            "tasks": [],
            "stages": [],
            "current_stage": [],
            "worker_results": ["__RESET__"],  # 触发自定义 reducer 清零
            "final_answer": "",
            "route": "",
        }

        # ── 滑动窗口压缩（与单 Agent 逻辑对齐）──
        raw_messages = state.get("messages") or []
        current_summary = state.get("summary", "")
        final_msgs, discarded_msgs = trim_context_messages(raw_messages, trigger_turns=40, keep_turns=10)

        if discarded_msgs:
            discarded_text = "\n".join([f"{m.type}: {m.content}" for m in discarded_msgs if m.content])
            summary_prompt = (
                f"你是一个负责维护 AI 工作台上下文的后台模块。\n\n"
                f"【现有的交接文档】\n{current_summary if current_summary else '暂无记录'}\n\n"
                f"【刚刚过去的旧对话】\n{discarded_text}\n\n"
                f"任务：请仔细阅读旧对话，提取出当前的对话语境和任务进度。\n"
                f"动作：将新进展与【现有的交接文档】进行无缝融合，输出一份最新的上下文摘要。\n"
                f"严格警告：只记录'我们在聊什么'、'解决了什么问题'、'得出了什么结论'等。绝对不要记录用户的静态偏好(如姓名、职业、爱好等)，这部分由其他模块负责！\n"
                f"要求：客观、精简，不要输出任何解释性废话，直接返回最新的记忆文本，总字数不要超过150字"
            )
            new_summary = llm.invoke([HumanMessage(content=summary_prompt)], config={"callbacks": []})
            current_summary = new_summary.content
            state_updates["summary"] = current_summary
            state_updates["messages"] = [RemoveMessage(id=m.id) for m in discarded_msgs if m.id]

        summary = current_summary
        prompt = (
            f"判断以下用户请求是否需要拆分为多个独立的并行子任务来执行。"
            f"{f'近期对话上下文：{summary}' if summary else ''}\n\n"
            f"只有明确包含多个独立目标时才回答 multi，例如：'分别调研A和B'、'同时处理这三个文件'。\n"
            f"闲聊、问候、单一问题、单一任务一律回答 chat。\n"
            f"只输出一个单词：chat 或 multi。\n\n"
            f"用户请求：{state['user_input']}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        route = "multi" if "multi" in response.content.strip().lower() else "chat"
        audit_logger.log_event(
            thread_id=_thread_id,
            event="system_action",
            content=f"[Multi-Agent] 意图路由: {route}"
        )
        state_updates["route"] = route
        return state_updates

    def route_decision(state: MultiState) -> Literal["chat_agent", "planner"]:
        return "chat_agent" if state["route"] == "chat" else "planner"

    # ─────────────── 普通对话节点 ───────────────
    def chat_agent_node(state: MultiState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        msgs = state.get("messages") or []
        summary = state.get("summary", "")
        sys_msg = SystemMessage(content=_base_sys_prompt(summary))
        non_sys = [m for m in msgs if not isinstance(m, SystemMessage)]

        # 判断是否是 tool 回调：最后一条消息是 ToolMessage 说明刚执行完工具，不需要追加 user_input
        is_tool_callback = bool(non_sys) and isinstance(non_sys[-1], ToolMessage)

        if is_tool_callback:
            response = llm_with_tools.invoke([sys_msg] + non_sys)
        else:
            human_msg = HumanMessage(content=state["user_input"])
            response = llm_with_tools.invoke([sys_msg] + non_sys + [human_msg])

        # 记录 tool_call 或 ai_message
        if response.tool_calls:
            for tc in response.tool_calls:
                audit_logger.log_event(
                    thread_id=_thread_id,
                    event="tool_call",
                    tool=tc["name"],
                    args=tc["args"]
                )
        elif response.content:
            audit_logger.log_event(
                thread_id=_thread_id,
                event="ai_message",
                content=response.content
            )

        if is_tool_callback:
            return {"messages": [response]}
        else:
            return {"messages": [human_msg, response]}

    def chat_tools_node(state: MultiState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        last_msg = state["messages"][-1]
        result = tool_node.invoke({"messages": [last_msg]})
        # 记录每条工具返回结果
        for msg in result["messages"]:
            audit_logger.log_event(
                thread_id=_thread_id,
                event="tool_result",
                tool=getattr(msg, "name", "unknown"),
                result_summary=str(msg.content)[:200]
            )
        return {"messages": result["messages"]}

    def chat_should_continue(state: MultiState) -> Literal["chat_tools", "chat_done"]:
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "chat_tools"
        return "chat_done"

    def chat_done_node(state: MultiState) -> dict:
        last_msg = state["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else ""
        return {"final_answer": answer}

    # ─────────────── Planner 节点 ───────────────
    def planner_node(state: MultiState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        profile = _load_profile()
        summary = state.get("summary", "")
        skills_section = ""
        if dynamic_skill_names:
            skills_list = "\n".join([f"- {n}" for n in dynamic_skill_names])
            skills_section = f"\n已加载的扩展技能（worker 可以使用）：\n{skills_list}\n"
        summary_section = f"\n【近期对话上下文】\n{summary}\n" if summary else ""

        prompt = (
            f"你是任务规划器。将用户请求拆解为子任务。\n"
            f"规则：数量 1-4 个，每个描述完整自包含。\n"
            f"如果子任务之间有先后依赖关系，在 depends_on 字段中填写前置任务的 id 列表；完全独立的任务 depends_on 为空列表。\n"
            f"只在子任务明确需要某技能时才提及技能。\n"
            f"{skills_section}{summary_section}"
            f"用户画像：{profile}\n\n"
            f"只输出 JSON 数组：[{{\"id\": 1, \"desc\": \"...\", \"depends_on\": []}}, ...]\n\n"
            f"用户请求：{state['user_input']}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            tasks = json.loads(content.strip())
            for t in tasks:
                if "depends_on" not in t:
                    t["depends_on"] = []
        except Exception:
            tasks = [{"id": 1, "desc": state["user_input"], "depends_on": []}]

        has_deps = any(t.get("depends_on") for t in tasks)
        stages_preview = _build_stages(tasks)
        mode = f"{len(stages_preview)}阶段" if has_deps else "并行"
        audit_logger.log_event(
            thread_id=_thread_id,
            event="system_action",
            content=f"[Multi-Agent] Planner 拆解出 {len(tasks)} 个子任务({mode}): {[t['desc'][:40] for t in tasks]}"
        )
        return {"tasks": tasks}

    # ─────────────── 审批节点 ───────────────
    def approval_node(state: MultiState) -> dict:
        tasks = state["tasks"]
        if len(tasks) <= 1:
            return {}
        stages = _build_stages(tasks)
        if len(stages) == 1:
            mode_label = "并行"
            plan_lines = "\n".join([f"  #{t['id']} {t['desc']}" for t in stages[0]])
        else:
            mode_label = "分阶段"
            lines = []
            for i, stage in enumerate(stages):
                lines.append(f"  [阶段{i+1}{'（并行）' if len(stage) > 1 else ''}]")
                for t in stage:
                    lines.append(f"    #{t['id']} {t['desc']}")
            plan_lines = "\n".join(lines)
        decision = interrupt({
            "type": "plan_approval",
            "plan": plan_lines,
            "task_count": len(tasks),
            "mode": mode_label,
        })
        if str(decision).strip().lower() in ["n", "no", "取消", "不", "cancel"]:
            feedback = interrupt({
                "type": "plan_feedback",
                "prompt": "请输入改进建议，我将重新规划："
            })
            new_input = f"{state['user_input']}\n\n【用户改进建议】{feedback}"
            return {"tasks": [], "user_input": new_input, "final_answer": "__replan__"}
        return {"final_answer": ""}

    # ─────────────── Worker 子图 ───────────────
    def worker_agent_node(state: WorkerState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        msgs = list(state["worker_messages"])

        # 如果是首次进入（只有 system + human），且有上一阶段产出，插入一条 context 消息
        if len(msgs) == 2 and state.get("prev_results"):
            context = "\n\n".join(state["prev_results"])
            msgs.insert(1, HumanMessage(
                content=f"【前序阶段产出，供本任务参考】\n{context}"
            ))

        response = llm_with_tools.invoke(msgs)
        if response.tool_calls:
            audit_logger.log_event(
                thread_id=_thread_id,
                event="system_action",
                content=f"[Worker #{state['task_id']}] 工具: {[tc['name'] for tc in response.tool_calls]}"
            )
        elif response.content:
            audit_logger.log_event(
                thread_id=_thread_id,
                event="ai_message",
                content=f"[Worker #{state['task_id']}] {response.content}"
            )
        return {"worker_messages": [response]}

    def worker_tools_node(state: WorkerState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        last_msg = state["worker_messages"][-1]
        result = tool_node.invoke({"messages": [last_msg]})
        for msg in result["messages"]:
            audit_logger.log_event(
                thread_id=_thread_id,
                event="tool_result",
                tool=getattr(msg, "name", "unknown"),
                result_summary=str(msg.content)[:200]
            )
        return {"worker_messages": result["messages"]}

    def worker_should_continue(state: WorkerState) -> Literal["tools", "done"]:
        last_msg = state["worker_messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "done"

    def worker_collect_result(state: WorkerState) -> dict:
        last_msg = state["worker_messages"][-1]
        result = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        return {"worker_results": [f"[子任务 {state['task_id']}：{state['task_desc'][:30]}...]\n{result}"]}

    def _build_stages(tasks: List[dict]) -> List[List[dict]]:
        """按依赖关系将任务分成多个阶段，同阶段内可并行，阶段间顺序执行"""
        task_map = {t["id"]: t for t in tasks}
        completed = set()
        remaining = list(tasks)
        stages = []
        while remaining:
            current = [t for t in remaining if all(d in completed for d in t.get("depends_on", []) if d in task_map)]
            if not current:
                current = list(remaining)  # 循环依赖兜底
            stages.append(current)
            for t in current:
                completed.add(t["id"])
            remaining = [t for t in remaining if t["id"] not in completed]
        return stages

    def stage_dispatch_node(state: MultiState) -> dict:
        """弹出下一个阶段，存入 current_stage，剩余存回 stages"""
        stages = state.get("stages") or []
        if not stages:
            stages = _build_stages(state["tasks"])
        current = stages[0]
        remaining = stages[1:]
        return {"current_stage": current, "stages": remaining}

    def dispatch_current_stage(state: MultiState) -> List[Send]:
        """将 current_stage 里的任务并行分发给 workers，经 Send() 传入上一阶段产出"""
        profile = _load_profile()
        skills_section = ""
        if dynamic_skill_names:
            skills_list = "\n".join([f"- {n}" for n in dynamic_skill_names])
            skills_section = f"\n【当前已加载的扩展技能】\n{skills_list}\n使用技能时先 mode='help' 读说明书。\n"
        sys_prompt = (
            "你是 ZenBot 的专注执行单元，负责完成分配给你的子任务。\n"
            "【重要】直接执行子任务，禁止追问、禁止要求澄清，遇到歧义自行做出合理假设后执行。\n"
            "完成后给出简洁结果摘要，所有文件操作限制在 office 目录内。\n"
            f"{skills_section}\n用户画像：{profile}"
        )
        # 上一阶段的产出通过状态字段传递，不拼进 system prompt
        prev_results = list(state.get("worker_results") or [])
        return [
            Send("worker", {
                "task_id": t["id"],
                "task_desc": t["desc"],
                "prev_results": prev_results,
                "worker_messages": [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=f"请完成以下子任务：{t['desc']}")
                ],
                "worker_results": []
            })
            for t in (state.get("current_stage") or [])
        ]

    worker_graph = StateGraph(WorkerState)
    worker_graph.add_node("agent", worker_agent_node)
    worker_graph.add_node("tools", worker_tools_node)
    worker_graph.add_node("collect", worker_collect_result)
    worker_graph.add_edge(START, "agent")
    worker_graph.add_conditional_edges("agent", worker_should_continue, {"tools": "tools", "done": "collect"})
    worker_graph.add_edge("tools", "agent")
    worker_graph.add_edge("collect", END)
    worker_subgraph = worker_graph.compile()

    # ─────────────── Aggregator 节点 ───────────────
    def aggregator_node(state: MultiState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        if state.get("final_answer") and state["final_answer"] != "__replan__":
            return {}

        combined = "\n\n".join(state.get("worker_results", []))

        # 还有下一阶段时，先检查本阶段是否成功，失败就提前终止
        if state.get("stages"):
            check_prompt = (
                f"以下是刚刚完成的阶段执行结果：\n{combined}\n\n"
                f"请判断这些结果是否代表【成功完成】了任务，还是遇到了【关键错误/失败】（例如：文件不存在、权限不足、找不到目标内容等）。\n"
                f"只输出一个单词：success 或 failure"
            )
            check = llm.invoke([HumanMessage(content=check_prompt)], config={"callbacks": []})
            if "failure" in check.content.strip().lower():
                # 本阶段失败，提前终止，直接告知用户
                fail_prompt = (
                    f"用户请求是：{state['user_input']}\n\n"
                    f"执行过程中遇到了问题，以下是失败的阶段结果：\n{combined}\n\n"
                    f"请如实告知用户任务未能完成的原因，并给出建议。"
                )
                response = llm.invoke([HumanMessage(content=fail_prompt)])
                audit_logger.log_event(
                    thread_id=_thread_id,
                    event="ai_message",
                    content=f"[阶段失败，提前终止] {response.content[:200]}"
                )
                return {
                    "final_answer": response.content,
                    "stages": [],  # 清空剩余阶段
                    # 写入 messages，后续 chat 轮次能看到这段历史
                    "messages": [
                        HumanMessage(content=state["user_input"]),
                        AIMessage(content=response.content)
                    ]
                }
            return {}  # 本阶段成功，继续下一阶段

        # 所有阶段完成，做最终汇总
        prompt = (
            f"用户的原始请求是：{state['user_input']}\n\n"
            f"以下是各子任务的执行结果：\n{combined}\n\n"
            "请整合成完整连贯的回复，直接回答用户，不要提及'子任务'、'worker'等内部概念。"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        audit_logger.log_event(
            thread_id=_thread_id,
            event="ai_message",
            content=f"[Multi-Agent 汇总] {response.content[:200]}"
        )
        return {
            "final_answer": response.content,
            # 写入 messages，后续 chat 轮次能看到这段历史
            "messages": [
                HumanMessage(content=state["user_input"]),
                AIMessage(content=response.content)
            ]
        }

    def aggregator_next(state: MultiState):
        """aggregator 后：还有阶段就继续 dispatch，否则结束"""
        if state.get("stages"):
            return "stage_dispatch"
        return END

    # ─────────────── 构建主图 ───────────────
    main_graph = StateGraph(MultiState)
    main_graph.add_node("router", router_node)
    main_graph.add_node("chat_agent", chat_agent_node)
    main_graph.add_node("chat_tools", chat_tools_node)
    main_graph.add_node("chat_done", chat_done_node)
    main_graph.add_node("planner", planner_node)
    main_graph.add_node("approval", approval_node)
    main_graph.add_node("stage_dispatch", stage_dispatch_node)
    main_graph.add_node("worker", worker_subgraph)
    main_graph.add_node("aggregator", aggregator_node)

    main_graph.add_edge(START, "router")
    main_graph.add_conditional_edges("router", route_decision, {"chat_agent": "chat_agent", "planner": "planner"})

    # 普通对话路径
    main_graph.add_conditional_edges("chat_agent", chat_should_continue, {"chat_tools": "chat_tools", "chat_done": "chat_done"})
    main_graph.add_edge("chat_tools", "chat_agent")
    main_graph.add_edge("chat_done", END)

    def approval_or_next(state: MultiState) -> Literal["planner", "aggregator", "stage_dispatch"]:
        if state.get("final_answer") == "__replan__":
            return "planner"
        if not state["tasks"]:
            return "aggregator"
        return "stage_dispatch"

    # 多任务路径
    main_graph.add_edge("planner", "approval")
    main_graph.add_conditional_edges("approval", approval_or_next, ["planner", "aggregator", "stage_dispatch"])
    main_graph.add_conditional_edges("stage_dispatch", dispatch_current_stage, ["worker"])
    main_graph.add_edge("worker", "aggregator")
    main_graph.add_conditional_edges("aggregator", aggregator_next, {"stage_dispatch": "stage_dispatch", END: END})

    app = main_graph.compile(checkpointer=checkpointer)
    return app
