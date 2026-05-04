# ZenBot 各节点 LLM 调用详解

> 最后更新：2026-05-03

本文档列出 ZenBot 多智能体管线中每个关键节点每次调用大模型时传入的完整信息，方便理解和调试。

---

## 数据流总览

```
用户输入
  │
  ▼
multi_subgraph_node（适配器：MainState → MultiAgentState）
  │  传入: user_input, history（最近3轮对话格式化）
  │
  ▼
planner_node ────────────────────────────────────────────── [LLM 调用 #1]
  │
  ▼
approval_node（人机交互，不调用 LLM）
  │
  ▼
stage_dispatch_node（纯逻辑，不调用 LLM）
  │
  ▼
dispatch_current_stage → Send() 创建 N 个 worker
  │
  ▼
worker_agent_node ───────────────────────────────────────── [LLM 调用 #2]
  │  ↕ （tool loop，最多 15 轮）
  ▼
worker_judge_node ───────────────────────────────────────── [LLM 调用 #3]
  │
  ▼
aggregator_node ─────────────────────────────────────────── [LLM 调用 #4]
  │
  ▼
memory_manager_node ─────────────────────────────────────── [LLM 调用 #5, #6]
  │
  ▼
END
```

---

## 1. planner_node — 任务规划器

**调用方式**：`llm.invoke([HumanMessage(content=prompt)])` — 单条消息，无 tools

**触发时机**：每轮用户输入后必经；aggregator 触发 `__replan__` 后再次进入

### Prompt 组成（按顺序拼接）

| # | 段落 | 内容 | 来源 |
|---|------|------|------|
| 1 | 角色 + 规则 | "你是任务规划器。将用户请求拆解为独立执行的子任务。" + 5 条硬编码规则 | 固定文本 |
| 2 | 时间基准 | "当前系统时间为：2026-05-03 14:30:22" | `datetime.now()` |
| 3 | 技能列表 | "已加载的扩展技能：\n- skill_a\n- skill_b" | `SKILLS_DIR` 动态扫描 |
| 4 | 用户画像 | `user_profile.md` 的全文内容（或"暂无记录"） | `workspace/memory/user_profile.md` |
| 5 | 长期记忆 | 最近 10 条 `memories/*.md` 的格式化内容 | `workspace/memory/memories/` |
| 6 | 对话历史 | 最近 3 轮对话（用户+AI），每条截断 500 字 | `MainState.messages` → `_format_history()` |
| 7 | JSON 格式要求 | 示例：`{"confidence": 1.0, "tasks": [], "direct_answer": "..."}` | 固定文本 |
| 8 | 用户请求 | 本轮用户原始输入 | `state['user_input']` |

### 完整 Prompt 模板

```
你是任务规划器。将用户请求拆解为独立执行的子任务。
规则：数量 1-4 个，每个描述完整自包含。
如果子任务之间有先后依赖关系，在 depends_on 字段中填写前置任务的 id 列表；完全独立的任务 depends_on 为空列表。
同时评估你对任务规划的置信度（0.0 到 1.0 的浮点数）。
【新增规则】对于纯闲聊或简单的明确问候（如'你好'、'谢谢'），请勿拆解任务。直接在 direct_answer 字段中给出回复，并让 tasks 为空数组。
【非常重要】绝不要额外分配一个用来'总结'或'撰写最终报告'的子任务！系统自身会执行全盘汇总分析。所有子任务只应该聚焦于实际的信息获取和操作执行。
【文件输出限制】除非用户在原始请求中明确要求"写入文件"、"生成文档"或"保存文件"，否则绝对不要分配生成或写入文件的任务。
只在子任务明确需要某技能时才提及技能。
【时间基准】当前系统时间为：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，如果你看到如"明天"、"下周"等时间词，请以此系统时间为基准推算日期直接分配在子任务描述里。

已加载的扩展技能（worker 可以使用）：
- skill_a
- skill_b

用户画像：{profile}

【长期记忆】以下是你之前记住的重要信息，规划任务时可参考：
{memories}

【最近对话历史】以下是前几轮的对话内容，用户可能会引用其中的信息：
{history}

只输出 JSON 对象，格式如下：
{"confidence": 1.0, "tasks": [], "direct_answer": "你好！有什么我可以帮你的？"}
或：{"confidence": 0.9, "tasks": [{"id": 1, "desc": "...", "depends_on": []}]}

用户请求：{user_input}
```

### 输出解析

- 解析 JSON，提取 `tasks`、`confidence`、`direct_answer`
- 解析失败时 fallback：`tasks = [{"id": 1, "desc": user_input, "depends_on": []}]`
- `tasks` 为空 + `direct_answer` 有值 → 直接作为 final_answer（闲聊快速路径）

---

## 2. worker_agent_node — 任务执行单元

**调用方式**：`llm_with_tools.invoke(msgs)` — 带工具绑定，支持多轮 tool call

**触发时机**：每个子任务各一个 worker 实例，并行执行

### 消息列表组成

| # | 消息类型 | 内容 | 来源 |
|---|----------|------|------|
| 1 | SystemMessage | 角色指令 + 规则 + 技能 + 画像 + 记忆 | `dispatch_current_stage` 构造 |
| 2 | HumanMessage（可选） | "【前序阶段产出】...必须基于以下结果完成本任务" | 仅当有前序 worker 结果时插入 |
| 3 | HumanMessage | "请完成以下子任务：{task_desc}" | 固定文本 + 子任务描述 |
| 4+ | 后续消息 | tool call / tool result / AI 回复 | tool loop 动态累积 |

### SystemMessage 完整内容

```
你是 ZenBot 的专注执行单元，负责完成分配给你的子任务。
【时间基准】当前系统时间为：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
【重要】直接执行子任务，禁止追问、禁止要求澄清，遇到歧义自行做出合理假设后执行。
【重要】如果收到【前序阶段产出】，必须优先基于这些已有结果完成任务，不要重复搜索或执行前序阶段已做过的工作。
【重要】如使用搜索引擎，只要能获取满足当前任务所需的基础核心信息即可，请立即停止继续搜索或调用工具，并直接输出结果。切勿过度搜索细节或无穷验证。如果查一次就已经能知道当前情况，就直接回复。
【文件生成限制】除非任务描述中明确指令"写文件"、"保存为文档"，否则绝对禁止调用任何工具创建、写入或修改文件，直接在聊天中以纯文本回复结果。
完成后给出简洁结果摘要，所有文件操作限制在 office 目录内。

【当前已加载的扩展技能】
- skill_a
- skill_b
使用技能时先 mode='help' 读说明书。

用户画像：{profile}

【长期记忆】你之前记住的信息（可参考，也可用 search_memory 工具搜索更多）：
{memories}
```

### 前序结果注入（条件触发）

仅在 worker 首次执行（`len(msgs) == 2`）且存在前序阶段结果时，在 SystemMessage 和 HumanMessage 之间插入：

```
【前序阶段产出，必须基于以下结果完成本任务，不要重复搜索】
{prev_results[0]}
{prev_results[1]}
...
```

### Tool Loop

- 最多 15 轮（`MAX_TOOL_LOOPS`）
- 第 13 轮时追加催促消息："工具调用即将达到上限，请立即停止调用工具，用已有信息完成任务并输出最终结果。"
- 可用工具：所有内置工具 + 动态技能工具（`web_search`、`read_office_file`、`write_office_file`、`execute_office_shell`、`save_memory`、`search_memory` 等）

---

## 3. worker_judge_node — 成功/失败判定

**调用方式**：`llm.invoke([HumanMessage(content=judge_prompt)], config={"callbacks": []})` — 无 callbacks（不记录日志）

**触发时机**：每个 worker 执行完毕后

### 判定逻辑

1. **快速跳过**：如果 worker 调用过任何工具（有 tool_calls），直接判定 success，不调用 LLM
2. **LLM 判定**：仅在 worker 没有调用工具时才调用 LLM 判断

### Prompt

```
以下是一个 AI 子任务执行后的输出结果：

{result_text}

请判断该结果是否代表任务执行【彻底失败】了。
只有以下情况才算 failure：明确的报错信息、异常堆栈、输出为空。
正常的回答、部分完成、有保留的结论都算 success。
只输出一个单词：success 或 failure
```

### 重试机制

- 判定为 failure 且 `retry_count < 3` → 重置 worker 消息为 `initial_messages`，重新执行
- 判定为 failure 且 `retry_count >= 3` → 标记失败，输出 "[执行失败] 经过 3 次重试仍未成功"
- 判定为 success → 放行，进入 collect

---

## 4. aggregator_node — 结果汇总

**调用方式**：`llm.invoke([HumanMessage(content=prompt)])` — 单条消息，无 tools

**触发时机**：所有 worker 完成后（每个阶段结束后都可能触发）

### 两种模式

#### 模式 A：阶段检查（还有后续阶段时）

先调用 LLM 判断当前阶段是否成功：

**阶段检查 Prompt**：
```
以下是刚刚完成的阶段执行结果：
{combined_results}

请判断这些结果是否代表【成功完成】了任务，还是遇到了【关键错误/失败】（例如：文件不存在、权限不足、找不到目标内容等）。
只输出一个单词：success 或 failure
```

- 判定为 failure → 再调用一次 LLM 生成用户友好的失败说明，提前终止（清空 `stages`）
- 判定为 success → 继续下一阶段

**失败说明 Prompt**：
```
用户请求是：{user_input}

执行过程中遇到了问题，以下是失败的阶段结果：
{combined_results}

请如实告知用户任务未能完成的原因，并给出建议。
```

#### 模式 B：最终汇总（所有阶段完成时）

**汇总 Prompt**：
```
用户的原始请求是：{user_input}

以下是各独立子任务的执行结果：
{combined_results}

你需要评估当前收集到的结果是否实质性地完成了用户的原始请求。
1. 如果已经成功完成，或者是由于超出能力/客观条件无法完成，请整合成完整连贯的回复直接回答用户。
2. 如果发现是因为先前的【执行计划有误】或方向不对导致任务失败，但你觉得换个方向或使用其他工具还有希望完成，请你输出：__replan__：[说明为什么失败以及应该怎么调整计划]

【重要】你的回复会被存入对话历史，后续对话中用户可能会引用你提到的内容。因此：
- 列举选项/方向/方案时，必须保留每个选项的名称和核心定义（不要只列编号）
- 关键数据、结论、专有名词必须保留，不要过度压缩
- 宁可稍长也不要丢失用户后续可能引用的细节
```

### 特殊路由

- 输出以 `__replan__` 开头 → 跳回 planner_node 重新规划，反思原因注入 `user_input`
- 单子任务 + 非 `__replan__` → 透传 worker 原始输出，跳过汇总 LLM 调用

---

## 5. memory_manager_node — 记忆管理（后置）

**调用方式**：`llm.invoke([HumanMessage(...)], config={"callbacks": []})` — 无 callbacks

**触发时机**：每轮对话结束后（multi_subgraph 之后）

### 两个独立的 LLM 调用

#### 调用 #1：滑动窗口压缩（仅 ≥40 轮时触发）

当对话消息数 ≥ 40 时，保留最新 10 轮，对被丢弃的旧消息生成摘要。

**压缩 Prompt**：
```
你是一个负责维护 AI 工作台上下文的后台模块。

【现有的交接文档】
{current_summary}  # 首次为空则显示"暂无记录"

【刚刚过去的旧对话】
{discarded_messages}  # 被滑动窗口丢弃的消息

任务：请仔细阅读旧对话，提取出当前的对话语境和任务进度。
动作：将新进展与【现有的交接文档】进行无缝融合，输出一份最新的上下文摘要。
严格警告：只记录'我们在聊什么'、'解决了什么问题'、'得出了什么结论'等。绝对不要记录用户的静态偏好(如姓名、职业、爱好等)，这部分由其他模块负责！
要求：客观、精简，不要输出任何解释性废话，直接返回最新的记忆文本，总字数不要超过150字
```

**输出**：更新 `MainState.summary`，被丢弃的消息从 `MainState.messages` 中移除

#### 调用 #2：自动记忆提取（每轮都尝试）

取最后一条 HumanMessage + AIMessage，判断是否有值得长期保存的信息。

**提取 Prompt**：
```
你是一个记忆提取模块。分析以下对话，判断是否有值得长期保存的信息。

【用户说】
{recent_human.content}

【AI 回复】
{recent_ai.content}

提取规则：
1. 只提取事实性、持久性的信息（用户偏好、决策、项目信息、技术方案等）
2. 不要提取临时信息（时间查询结果、一次性计算、闲聊寒暄）
3. 不要提取 user_profile 已有的信息（姓名、称呼等）
4. 不要提取对话流程信息（如'用户让我搜索了XX'）

如果值得保存，输出 JSON：
{"save": true, "content": "简洁的记忆内容（1-3句）", "category": "fact/preference/decision/project/technical/general", "keywords": "逗号分隔关键词"}
如果没有值得保存的信息：
{"save": false}
```

**输出**：JSON 解析成功且 `save=true` → 调用 `save_memory_to_disk()` 写入 `memories/` 目录

---

## 附：multi_subgraph_node — 状态适配器

**不调用 LLM**，纯数据转换。

### MainState → MultiAgentState 映射

```python
sub_input = {
    "user_input": state["user_input"],          # 原样传入
    "tasks": [],                                 # 空，由 planner 填充
    "stages": [],                                # 空，由 stage_dispatch 填充
    "current_stage": [],                         # 空
    "worker_results": [],                        # 空，由 worker 追加
    "final_answer": "",                          # 空，由 aggregator 填充
    "confidence": 0.0,                           # 由 planner 填充
    "history": _format_history(state["messages"]) # 最近3轮对话格式化
}
```

### `_format_history()` 逻辑

- 遍历 `MainState.messages`，按 Human+AI 配对分组
- 取最后 3 轮（`max_turns=3`）
- 每条消息截断 500 字
- 输出格式：`[用户] ...\n[AI] ...\n[用户] ...\n[AI] ...`

### MultiAgentState → MainState 回写

```python
return {
    "messages": [AIMessage(content=final_answer)],  # aggregator 输出存入对话历史
    "user_input": "",                                 # 清空
    "final_answer": "",                               # 清空
}
```

---

## 附：完整 LLM 调用次数估算

| 场景 | LLM 调用次数 | 说明 |
|------|-------------|------|
| 闲聊（direct_answer） | 1 | planner 直接回复，跳过 worker/aggregator |
| 单任务 + 高置信 | 2~3 | planner + worker + (judge 跳过) + aggregator 透传 |
| 单任务 + 普通 | 3~4 | planner + worker + judge + aggregator |
| 多任务并行 | 3+N+1 | planner + N 个 worker（并行）+ aggregator |
| 多阶段 + replan | 翻倍 | 上述 + 再来一轮 planner → workers → aggregator |
| 记忆提取（每轮） | +1~2 | 压缩（条件触发）+ 提取（每轮） |
