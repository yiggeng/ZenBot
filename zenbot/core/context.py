import operator
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages


class MainState(TypedDict):
    """主图持久化状态：跨轮次保留的字段"""
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    user_input: str
    final_answer: str


class MultiAgentState(TypedDict):
    """多智能体子图的内部状态：仅在 multi 路径内存活"""
    user_input: str
    tasks: List[dict]
    stages: List[List[dict]]
    current_stage: List[dict]
    worker_results: Annotated[List[str], operator.add]  # 并行 worker 追加合并
    final_answer: str
    confidence: float       # planner 输出的置信度，用于 approval 节点判断是否跳过
    history: str            # 最近几轮对话历史（从 MainState.messages 格式化后传入）
    profile: str            # 用户画像（入口加载一次，下游节点复用）
    memories: str           # 长期记忆摘要（入口加载一次，下游节点复用）


def trim_context_messages(
    messages: list[BaseMessage],
    trigger_turns: int = 8,
    keep_turns: int = 4,
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    # 按完整用户回合裁剪：一个回合从 HumanMessage 开始，包含后续所有 AI/Tool 消息
    first_system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_msgs:
        return ([first_system] if first_system else []), []

    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    for msg in non_system_msgs:
        if isinstance(msg, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            if current_turn:
                current_turn.append(msg)

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
