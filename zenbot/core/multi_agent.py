import json
import operator
import os
from typing import Annotated, List, Literal, TypedDict
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage, RemoveMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Send, interrupt

from .config import MEMORY_DIR
from .context import MainState, MultiAgentState, trim_context_messages
from .logger import audit_logger
from .provider import get_provider
from .skill_loader import load_dynamic_skills
from .tools.builtins import BUILTIN_TOOLS


# ─────────────────────────── Worker State ───────────────────────────

MAX_TOOL_LOOPS = 15

class WorkerState(TypedDict):
    task_id: int
    task_desc: str
    prev_results: List[str]
    initial_messages: List[BaseMessage]   # 保存初始 prompt，用于重试时重置
    retry_count: int
    should_retry: bool
    tool_loop_count: int                  # 当前 tool loop 轮数，防无限循环
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

    CONFIDENCE_THRESHOLD = 0.7

    def _load_profile() -> str:
        profile_path = os.path.join(MEMORY_DIR, "user_profile.md")
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    return content
        return "暂无记录"

    # ─────────────── 记忆管理节点（后置，对话结束前执行）───────────────
    def memory_manager_node(state: MainState, config: RunnableConfig) -> dict:
        """滑动窗口压缩：在对话结束后异步压缩旧消息，不阻塞路由判断"""
        raw_messages = state.get("messages") or []
        current_summary = state.get("summary", "")
        final_msgs, discarded_msgs = trim_context_messages(raw_messages, trigger_turns=40, keep_turns=10)

        if not discarded_msgs:
            return {}

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
        return {
            "summary": new_summary.content,
            "messages": [RemoveMessage(id=m.id) for m in discarded_msgs if m.id],
        }

    # ─────────────── Worker 子图节点 ───────────────
    def worker_agent_node(state: WorkerState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        msgs = list(state["worker_messages"])

        if len(msgs) == 2 and state.get("prev_results"):
            context = "\n\n".join(state["prev_results"])
            msgs.insert(1, HumanMessage(
                content=f"【前序阶段产出，必须基于以下结果完成本任务，不要重复搜索】\n{context}"
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
        new_count = state.get("tool_loop_count", 0) + 1
        msgs_update = list(result["messages"])
        # 快到上限时追加催促，让下一轮 LLM 直接输出总结
        if new_count >= MAX_TOOL_LOOPS - 2:
            msgs_update.append(HumanMessage(
                content="工具调用即将达到上限，请立即停止调用工具，用已有信息完成任务并输出最终结果。"
            ))
        return {"worker_messages": msgs_update, "tool_loop_count": new_count}

    def worker_should_continue(state: WorkerState) -> Literal["tools", "done"]:
        last_msg = state["worker_messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "done"

    def worker_collect_result(state: WorkerState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        last_msg = state["worker_messages"][-1]
        result = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # 如果最后一条消息没有 content（被 tool loop 截断），用最后一条 ToolMessage 的结果
        if not result.strip():
            for msg in reversed(state["worker_messages"]):
                if isinstance(msg, ToolMessage) and msg.content:
                    result = f"[工具调用已达上限，以下为最后获取的结果]\n{msg.content}"
                    break
            if not result.strip():
                result = "[任务未完成：工具调用次数已达上限，未能生成最终结果]"
        audit_logger.log_event(
            thread_id=_thread_id,
            event="ai_message",
            content=f"[Worker #{state['task_id']}] 最终结果:\n{result}"
        )
        return {"worker_results": [f"[子任务 {state['task_id']}：{state['task_desc'][:30]}...]\n{result}"]}

    def worker_judge_node(state: WorkerState, config: RunnableConfig) -> dict:
        """判断 worker 最新结果是否成功；失败时重试，最多 3 次"""
        last_msg = state["worker_messages"][-1]
        result_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # 如果 worker 调用过工具，大概率是成功的，跳过 judge 直接放行
        tool_calls_exist = any(
            hasattr(m, "tool_calls") and m.tool_calls
            for m in state["worker_messages"]
        )
        if tool_calls_exist:
            return {"should_retry": False}

        judge_prompt = (
            f"以下是一个 AI 子任务执行后的输出结果：\n\n{result_text}\n\n"
            f"请判断该结果是否代表任务执行【彻底失败】了。\n"
            f"只有以下情况才算 failure：明确的报错信息、异常堆栈、输出为空。\n"
            f"正常的回答、部分完成、有保留的结论都算 success。\n"
            f"只输出一个单词：success 或 failure"
        )
        check = llm.invoke([HumanMessage(content=judge_prompt)], config={"callbacks": []})
        is_failure = "failure" in check.content.strip().lower()

        retry_count = state.get("retry_count", 0)
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")

        if is_failure and retry_count < 3:
            audit_logger.log_event(
                thread_id=_thread_id,
                event="system_action",
                content=f"[Worker #{state['task_id']}] 执行失败，第 {retry_count + 1} 次重试"
            )
            initial = list(state.get("initial_messages") or [])
            remove_ops = [RemoveMessage(id=m.id) for m in state["worker_messages"] if getattr(m, "id", None)]
            return {
                "retry_count": retry_count + 1,
                "should_retry": True,
                "tool_loop_count": 0,
                "worker_messages": remove_ops + initial,
            }

        if is_failure and retry_count >= 3:
            audit_logger.log_event(
                thread_id=_thread_id,
                event="system_action",
                content=f"[Worker #{state['task_id']}] 重试 3 次后仍失败，标记为失败"
            )
            return {
                "should_retry": False,
                "worker_results": [
                    f"[子任务 {state['task_id']}：{state['task_desc'][:30]}...]\n"
                    f"[执行失败] 经过 3 次重试仍未成功。最后一次输出：{result_text[:300]}"
                ]
            }

        return {"should_retry": False}

    def worker_judge_route(state: WorkerState) -> Literal["agent", END]:
        if state.get("should_retry"):
            return "agent"
        return END

    worker_graph = StateGraph(WorkerState)
    worker_graph.add_node("agent", worker_agent_node)
    worker_graph.add_node("tools", worker_tools_node)
    worker_graph.add_node("collect", worker_collect_result)
    worker_graph.add_node("judge", worker_judge_node)
    worker_graph.add_edge(START, "agent")
    worker_graph.add_conditional_edges("agent", worker_should_continue, {"tools": "tools", "done": "collect"})
    worker_graph.add_edge("tools", "agent")
    worker_graph.add_edge("collect", "judge")
    worker_graph.add_conditional_edges("judge", worker_judge_route, {"agent": "agent", END: END})
    worker_subgraph = worker_graph.compile()

    # ─────────────── 多智能体子图节点 ───────────────

    def _build_stages(tasks: List[dict]) -> List[List[dict]]:
        task_map = {t["id"]: t for t in tasks}
        completed = set()
        remaining = list(tasks)
        stages = []
        while remaining:
            current = [
                t for t in remaining
                if all(d in completed for d in t.get("depends_on", []) if d in task_map)
            ]
            if not current:
                current = list(remaining)  # 循环依赖兜底
            stages.append(current)
            for t in current:
                completed.add(t["id"])
            remaining = [t for t in remaining if t["id"] not in completed]
        return stages

    def planner_node(state: MultiAgentState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        profile = _load_profile()
        skills_section = ""
        if dynamic_skill_names:
            skills_list = "\n".join([f"- {n}" for n in dynamic_skill_names])
            skills_section = f"\n已加载的扩展技能（worker 可以使用）：\n{skills_list}\n"

        prompt = (
            f"你是任务规划器。将用户请求拆解为独立执行的子任务。\n"
            f"规则：数量 1-4 个，每个描述完整自包含。\n"
            f"如果子任务之间有先后依赖关系，在 depends_on 字段中填写前置任务的 id 列表；完全独立的任务 depends_on 为空列表。\n"
            f"同时评估你对任务规划的置信度（0.0 到 1.0 的浮点数）。\n"
            f"【新增规则】对于纯闲聊或简单的明确问候（如'你好'、'谢谢'），请勿拆解任务。直接在 direct_answer 字段中给出回复，并让 tasks 为空数组。\n"
            f"【非常重要】绝不要额外分配一个用来'总结'或'撰写最终报告'的子任务！系统自身会执行全盘汇总分析。所有子任务只应该聚焦于实际的信息获取和操作执行。\n"
            f"【文件输出限制】除非用户在原始请求中明确要求“写入文件”、“生成文档”或“保存文件”，否则绝对不要分配生成或写入文件的任务。\n"
            f"只在子任务明确需要某技能时才提及技能。\n"
            f"【时间基准】当前系统时间为：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，如果你看到如“明天”、“下周”等时间词，请以此系统时间为基准推算日期直接分配在子任务描述里。\n"
            f"{skills_section}"
            f"用户画像：{profile}\n\n"
            f"只输出 JSON 对象，格式如下：\n"
            f'{{"confidence": 1.0, "tasks": [], "direct_answer": "你好！有什么我可以帮你的？"}}\n'
            f'或：{{"confidence": 0.9, "tasks": [{{"id": 1, "desc": "...", "depends_on": []}}]}}\n\n'
            f"用户请求：{state['user_input']}"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        confidence, direct_answer = 0.5, ""
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
            tasks = parsed.get("tasks", [])
            confidence = float(parsed.get("confidence", 0.5))
            direct_answer = parsed.get("direct_answer", "")
            for t in tasks:
                if "depends_on" not in t:
                    t["depends_on"] = []
        except Exception:
            tasks = [{"id": 1, "desc": state["user_input"], "depends_on": []}]

        has_deps = any(t.get("depends_on") for t in tasks)
        stages_preview = _build_stages(tasks)
        mode = f"{len(stages_preview)}阶段" if has_deps else ("直接回复" if not tasks else "并行")
        audit_logger.log_event(
            thread_id=_thread_id,
            event="system_action",
            content=f"[Planner] 置信度: {confidence:.2f}, 拆解出 {len(tasks)} 个子任务({mode}): {[t['desc'][:40] for t in tasks]}"
        )
        updates = {"tasks": tasks, "confidence": confidence}
        if not tasks and direct_answer:
            updates["final_answer"] = direct_answer
        return updates

    def approval_node(state: MultiAgentState) -> dict:
        tasks = state["tasks"]
        confidence = state.get("confidence", 0.0)
        # 单任务 + 高置信 → 跳过 approval，直接执行
        if len(tasks) <= 1 and confidence >= CONFIDENCE_THRESHOLD:
            audit_logger.log_event(
                thread_id="",
                event="system_action",
                content=f"[Approval] 单任务 + 高置信({confidence:.2f})，跳过确认直接执行"
            )
            return {}
        
        # 否则（多任务或低置信单任务） → 走 approval interrupt
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

    def stage_dispatch_node(state: MultiAgentState) -> dict:
        stages = state.get("stages") or []
        if not stages:
            stages = _build_stages(state["tasks"])
        current = stages[0]
        remaining = stages[1:]
        return {"current_stage": current, "stages": remaining}

    def dispatch_current_stage(state: MultiAgentState) -> List[Send]:
        profile = _load_profile()
        skills_section = ""
        if dynamic_skill_names:
            skills_list = "\n".join([f"- {n}" for n in dynamic_skill_names])
            skills_section = f"\n【当前已加载的扩展技能】\n{skills_list}\n使用技能时先 mode='help' 读说明书。\n"
        sys_prompt = (
            f"你是 ZenBot 的专注执行单元，负责完成分配给你的子任务。\n"
            f"【时间基准】当前系统时间为：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            "【重要】直接执行子任务，禁止追问、禁止要求澄清，遇到歧义自行做出合理假设后执行。\n"
            "【重要】如果收到【前序阶段产出】，必须优先基于这些已有结果完成任务，不要重复搜索或执行前序阶段已做过的工作。\n"
            "【重要】如使用搜索引擎，只要能获取满足当前任务所需的基础核心信息即可，请立即停止继续搜索或调用工具，并直接输出结果。切勿过度搜索细节或无穷验证。如果查一次就已经能知道当前情况，就直接回复。\n"
            "【文件生成限制】除非任务描述中明确指令“写文件”、“保存为文档”，否则绝对禁止调用任何工具创建、写入或修改文件，直接在聊天中以纯文本回复结果。\n"
            "完成后给出简洁结果摘要，所有文件操作限制在 office 目录内。\n"
            f"{skills_section}\n用户画像：{profile}"
        )
        prev_results = list(state.get("worker_results") or [])
        return [
            Send("worker", {
                "task_id": t["id"],
                "task_desc": t["desc"],
                "prev_results": prev_results,
                "initial_messages": [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=f"请完成以下子任务：{t['desc']}")
                ],
                "retry_count": 0,
                "tool_loop_count": 0,
                "worker_messages": [
                    SystemMessage(content=sys_prompt),
                    HumanMessage(content=f"请完成以下子任务：{t['desc']}")
                ],
                "worker_results": []
            })
            for t in (state.get("current_stage") or [])
        ]

    def aggregator_node(state: MultiAgentState, config: RunnableConfig) -> dict:
        _thread_id = config.get("configurable", {}).get("thread_id", "zenbot_main")
        if state.get("final_answer") and state["final_answer"] != "__replan__":
            return {}

        combined = "\n\n".join(state.get("worker_results", []))

        if state.get("stages"):
            check_prompt = (
                f"以下是刚刚完成的阶段执行结果：\n{combined}\n\n"
                f"请判断这些结果是否代表【成功完成】了任务，还是遇到了【关键错误/失败】（例如：文件不存在、权限不足、找不到目标内容等）。\n"
                f"只输出一个单词：success 或 failure"
            )
            check = llm.invoke([HumanMessage(content=check_prompt)], config={"callbacks": []})
            if "failure" in check.content.strip().lower():
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
                    "stages": [],
                }
            return {}

        if len(state.get("tasks", [])) == 1 and not state.get("stages") and "failure" not in combined.lower():
            results = state.get("worker_results", [])
            if results:
                parts = results[0].split("\n", 1)
                raw_answer = parts[1].strip() if len(parts) > 1 else results[0]
                audit_logger.log_event(
                    thread_id=_thread_id,
                    event="ai_message",
                    content=f"[Aggregator] 透传 Worker 输出（跳过重写）"
                )
                return {"final_answer": raw_answer}

        prompt = (
            f"用户的原始请求是：{state['user_input']}\n\n"
            f"以下是各独立子任务的执行结果：\n{combined}\n\n"
            f"你需要评估当前收集到的结果是否实质性地完成了用户的原始请求。\n"
            f"1. 如果已经成功完成，或者是由于超出能力/客观条件无法完成，请整合成完整连贯的回复直接回答用户。\n"
            f"2. 如果发现是因为先前的【执行计划有误】或方向不对导致任务失败，但你觉得换个方向或使用其他工具还有希望完成，请你输出：__replan__：[说明为什么失败以及应该怎么调整计划]"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        if content.startswith("__replan__"):
            feedback = content.replace("__replan__", "").strip(":： \n")
            audit_logger.log_event(
                thread_id=_thread_id,
                event="system_action",
                content=f"[Aggregator 触发重新规划] 反思建议：{feedback}"
            )
            new_input = f"{state['user_input']}\n\n【前次执行失败反思】：{feedback}"
            return {"user_input": new_input, "final_answer": "__replan__", "tasks": []}

        audit_logger.log_event(
            thread_id=_thread_id,
            event="ai_message",
            content=f"[Aggregator] {content[:200]}"
        )
        return {"final_answer": content}

    def aggregator_next(state: MultiAgentState):
        if state.get("final_answer") == "__replan__":
            return "planner"
        if state.get("stages"):
            return "stage_dispatch"
        return END

    def approval_or_next(state: MultiAgentState) -> Literal["planner", "aggregator", "stage_dispatch"]:
        if state.get("final_answer") == "__replan__":
            return "planner"
        if not state["tasks"]:
            return "aggregator"
        return "stage_dispatch"

    # ─────────────── 构建多智能体子图 ───────────────
    multi_graph = StateGraph(MultiAgentState)
    multi_graph.add_node("planner", planner_node)
    multi_graph.add_node("approval", approval_node)
    multi_graph.add_node("stage_dispatch", stage_dispatch_node)
    multi_graph.add_node("worker", worker_subgraph)
    multi_graph.add_node("aggregator", aggregator_node)

    multi_graph.add_edge(START, "planner")
    multi_graph.add_edge("planner", "approval")
    multi_graph.add_conditional_edges("approval", approval_or_next, ["planner", "aggregator", "stage_dispatch"])
    multi_graph.add_conditional_edges("stage_dispatch", dispatch_current_stage, ["worker"])
    multi_graph.add_edge("worker", "aggregator")
    multi_graph.add_conditional_edges("aggregator", aggregator_next, {"stage_dispatch": "stage_dispatch", "planner": "planner", END: END})

    compiled_multi = multi_graph.compile()

    # ─────────────── 多智能体子图适配节点（MainState <-> MultiAgentState）───────────────
    def multi_subgraph_node(state: MainState, config: RunnableConfig) -> dict:
        """调用多智能体子图，将结果冒泡回主图 messages"""
        sub_input: MultiAgentState = {
            "user_input": state["user_input"],
            "tasks": [],
            "stages": [],
            "current_stage": [],
            "worker_results": [],
            "final_answer": "",
            "confidence": 0.0,
        }
        # 子图同步执行（invoke），内部 interrupt 会向上传播
        sub_result = compiled_multi.invoke(sub_input, config)
        answer = sub_result.get("final_answer", "")
        updates: dict = {"final_answer": answer}
        if answer and answer != "__replan__":
            updates["messages"] = [
                HumanMessage(content=state["user_input"]),
                AIMessage(content=answer),
            ]
        return updates

    # ─────────────── 构建主图 ───────────────
    main_graph = StateGraph(MainState)
    main_graph.add_node("multi_subgraph", multi_subgraph_node)
    main_graph.add_node("memory_manager", memory_manager_node)

    main_graph.add_edge(START, "multi_subgraph")
    main_graph.add_edge("multi_subgraph", "memory_manager")
    main_graph.add_edge("memory_manager", END)

    app = main_graph.compile(checkpointer=checkpointer)
    return app
