from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages


def _merge_worker_results(existing: List[str], new: List[str]) -> List[str]:
    """worker_results 的自定义 reducer：
    - 如果 new 是 ['__RESET__']，则清零返回空列表
    - 否则追加（支持多 worker 并行合并）
    """
    if new == ["__RESET__"]:
        return []
    return existing + new


class AgentState(TypedDict):
    """单 Agent 模式的状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


class MultiState(TypedDict):
    """多 Agent 模式的状态（messages 同样使用 add_messages 增量追加）"""
    # ── 共有字段（与 AgentState 对齐）──
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    # ── 多 Agent 专用字段 ──
    user_input: str
    route: str
    tasks: List[dict]
    stages: List[List[dict]]
    current_stage: List[dict]
    worker_results: Annotated[List[str], _merge_worker_results]
    final_answer: str

def trim_context_messages(messages: list[BaseMessage], trigger_turns: int = 8, keep_turns: int = 4) -> tuple[list[BaseMessage], list[BaseMessage]]:
    # 按照完整用户回合来裁剪上下文：即 一个会从从HumanMessage开始，直到下一个HumanMessage结束，会把AIMessage、tool_calls、ToolMessage一并保留
    first_system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_msgs:
        return ([first_system] if first_system else []), []
    
    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    # 遍历非系统信息，按回合进行分组
    for msg in non_system_msgs:
        if isinstance(msg, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            if current_turn:
                current_turn.append(msg)
    
    # 保存最后一个回合
    if current_turn:
        turns.append(current_turn)

    total_turns = len(turns)

    if total_turns < trigger_turns:
        final_messages = ([first_system] if first_system else []) + non_system_msgs
        return final_messages, []
    
    recent_turns = turns[-keep_turns:]
    discarded_turns = turns[:-keep_turns]

    final_messages: list[BaseMessage] = []
    if first_system:
        final_messages.append(first_system)
    for turn in recent_turns:
        final_messages.extend(turn)

    discarded_messages: list[BaseMessage] = []
    for turn in discarded_turns:
        discarded_messages.extend(turn)

    return final_messages, discarded_messages

    
