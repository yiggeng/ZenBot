# ZenBot 各节点 LLM 调用详解

> 最后更新：2026-05-14

本文档列出 ZenBot 两种管线模式（多智能体 / 深度研究）中每个关键节点每次调用大模型时传入的完整信息，方便理解和调试。

---

## 数据流总览

ZenBot 根据 `mode` 参数路由到两条不同的管线：

### 管线 A：多智能体模式（mode=multi_agent）

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

### 管线 B：深度研究模式（mode=deep_research）

```
用户输入
  │
  ▼
deep_research_subgraph
  │
  ▼
generate_query_node ─────────────────────────────────────── [DR-LLM #1]
  │  生成搜索查询列表
  ▼
_dispatch_queries → Send() 并行分发 N 个查询
  │
  ▼
web_research_node ───────────────────────────────────────── [DR-LLM #2]
  │  Tavily 搜索 + LLM 摘要（每个查询并行执行）
  ▼
reflection_node ─────────────────────────────────────────── [DR-LLM #3]
  │  判断研究是否充分，生成后续查询
  ▼
evaluate_research（条件路由）
  │  ├─ 不充分 & 未达上限 → 回到 web_research（循环，默认最多 2 轮）
  │  └─ 充分 or 达上限 → 进入质量保证流水线
  ▼
assess_content_quality_node ─────────────────────────────── [DR-LLM #4]
  │
  ▼
verify_facts_node ───────────────────────────────────────── [DR-LLM #5]
  │
  ▼
assess_relevance_node ───────────────────────────────────── [DR-LLM #6]
  │
  ▼
optimize_summary_node ───────────────────────────────────── [DR-LLM #7]
  │
  ▼
generate_verification_report_node（纯文本组合，不调用 LLM）
  │
  ▼
finalize_answer_node（写入 reports/ 目录，不调用 LLM）
  │
  ▼
memory_manager_node ─────────────────────────────────────── [LLM 调用 #5, #6]
  │
  ▼
END
```

> **注意**：两条管线最终都经过 `memory_manager_node` 进行记忆压缩和提取。

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

## 6. Deep Research 管线 — 7 节点 LLM 调用详解

深度研究管线是 ZenBot 的第二条执行路径（`mode=deep_research`），采用 "查询生成 → 网络搜索 → 反思循环 → 质量保证流水线 → 报告生成" 的架构。所有 LLM 调用通过 `_make_llm(temperature)` 辅助函数创建，使用系统配置的 provider。

### 状态模型

深度研究生效有两个 TypedDict 状态：

- **DeepResearchState**：持久化状态，在 `graph.py` 中定义。关键字段：`user_input`、`search_query`、`web_research_result`（`add_messages` reducer）、`sources_gathered`、`is_sufficient`、`knowledge_gap`、`follow_up_queries`、`research_loop_count`、`content_quality`、`fact_verification`、`relevance_assessment`、`summary_optimization`、`quality_enhanced_summary`、`verification_report`、`final_answer`
- **WebSearchState**：每个 `web_research` worker 的临时状态，包含 `search_query` 和 `id`

### 图拓扑

```
START → generate_query → [并行 web_research*] → reflection → evaluate_research
                                                      │
                          ┌───────────────┐          │
                          │ 不充分 + 未达上限 → web_research（循环）
                          │ 充分 or 达上限 → assess_content_quality
                          └───────────────┘
assess_content_quality → verify_facts → assess_relevance → optimize_summary
  → generate_verification_report（无LLM） → finalize_answer（无LLM） → END
```

---

### 6.1 generate_query_node — 搜索查询生成器 [DR-LLM #1]

**调用方式**：`llm.with_structured_output(SearchQueryList).ainvoke(formatted_prompt)` — 结构化 JSON 输出，无 tools

**触发时机**：START 后第一个节点，总是执行

**温度**：1.0（高创造力，鼓励查询多样化）

**SearchQueryList Pydantic 模型**：
```python
class SearchQueryList(BaseModel):
    rationale: str        # 查询相关性说明
    query: List[str]      # 搜索查询列表（1~N 个）
```

**Prompt 模板**（来自 `prompts.py:query_writer_instructions`）：
```
你的目标是生成复杂且多样化的网络搜索查询。这些查询用于高级自动化网络研究工具，该工具能够分析复杂结果、跟踪链接并综合信息。

指令：
- 始终优先使用单个搜索查询，只有在原始问题要求多个方面或元素且一个查询不够时才添加另一个查询。
- 每个查询应专注于原始问题的一个特定方面。
- 不要产生超过 {number_queries} 个查询。
- 查询应该多样化，如果主题广泛，生成超过1个查询。
- 不要生成多个相似的查询，1个就足够了。
- 查询应确保收集最新信息。当前日期是 {current_date}。

格式：
- 将您的回复格式化为具有所有两个确切键的JSON对象：
   - "rationale": 为什么这些查询相关的简要解释
   - "query": 搜索查询列表

示例：

主题：去年苹果股票收入增长和购买iPhone的人数增长哪个更多
{"rationale": "...", "query": ["苹果2024财年总收入增长", "iPhone 2024财年单位销售增长", "苹果2024财年股价增长"]}

上下文：{research_topic}
```

**占位符填充**：
| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{current_date}` | `get_current_date()` → `datetime.now()` | `2026年05月14日` |
| `{research_topic}` | `state["user_input"]` | 用户原始问题 |
| `{number_queries}` | `state["initial_search_query_count"]` 或环境变量 `DR_INITIAL_QUERIES`（默认 3） | `3` |

**输出**：`result.query` 存入 `state["search_query"]` 和 `state["generated_queries"]`，然后通过 `_dispatch_queries()` 并行 Send 到 `web_research` 节点

---

### 6.2 web_research_node — 网络搜索与摘要 [DR-LLM #2]

**调用方式**：`llm.ainvoke(analysis_prompt)` — 纯文本输出，无结构化输出，无 tools

**触发时机**：
1. `generate_query` 后通过 `_dispatch_queries` 并行触发（每个查询一个 worker）
2. `reflection` 判定研究不充分时再次触发（循环）

**温度**：0.0（确定性的、事实性摘要）

**执行流程**：
1. 调用 Tavily API（`search_depth="advanced"`, `max_results=5`）
2. 将搜索结果格式化为文本块
3. 将搜索结果 + 指令 prompt 拼接发给 LLM
4. LLM 返回带引用链接的综合摘要
5. 将 URL 替换为短引用标记（`[1]`、`[2]` 等）

**Prompt 模板**（来自 `prompts.py:web_searcher_instructions`）：
```
进行有针对性的Google搜索，收集关于"{research_topic}"的最新、可信信息，并将其合成为可验证的文本内容。

指令：
- 查询应确保收集最新信息。当前日期是 {current_date}。
- 进行多次、多样化的搜索以收集全面信息。
- 整合关键发现，同时仔细跟踪每个具体信息的来源。
- 输出应该是基于搜索发现的结构良好的摘要或报告。
- 只包含在搜索结果中找到的信息，不要编造任何信息。
- **重要：在引用信息时，请使用markdown链接格式 [引用文本](URL) 来标注来源。**
- **每当提到具体事实、数据或观点时，都应该包含相应的引用链接。**

研究主题：
{research_topic}
```

**实际发送给 LLM 的完整消息**：
```
{上述模板填充后的内容}

搜索结果：
Source 1: {title}
URL: {url}
Content: {content}

Source 2: {title}
URL: {url}
Content: {content}

...（最多 5 个结果）

请分析这些搜索结果并提供带有引用的综合摘要。请用中文回答。
```

**占位符填充**：
| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{current_date}` | `get_current_date()` | `2026年05月14日` |
| `{research_topic}` | `state["search_query"]`（单个查询字符串） | `苹果2024财年总收入增长` |
| 搜索结果 | Tavily API 返回（最多 5 条） | 标题 + URL + 正文片段 |

**输出**：
- `state["web_research_result"]`：追加 LLM 摘要文本（`add_messages` reducer）
- `state["sources_gathered"]`：追加来源字典列表 `[{title, url, content, short_url, value, label}]`

---

### 6.3 reflection_node — 研究反思 [DR-LLM #3]

**调用方式**：`llm.with_structured_output(Reflection).ainvoke(formatted_prompt)` — 结构化 JSON 输出

**触发时机**：每次 `web_research` 完成后（所有并行 worker 结果收集完毕后）

**温度**：1.0（高创造力，有利于发现知识缺口）

**Reflection Pydantic 模型**：
```python
class Reflection(BaseModel):
    is_sufficient: bool              # 研究是否足以回答问题
    knowledge_gap: str               # 知识缺口描述
    follow_up_queries: List[str]     # 后续搜索查询
```

**Prompt 模板**（来自 `prompts.py:reflection_instructions`）：
```
你是一名专业的研究助手，正在分析关于"{research_topic}"的摘要。

指令：
- 识别知识差距或需要深入探索的领域，并生成后续查询（1个或多个）。
- 如果提供的摘要足以回答用户的问题，则不要生成后续查询。
- 如果存在知识差距，生成有助于扩展理解的后续查询。
- 专注于未充分涵盖的技术细节、实施具体内容或新兴趋势。

要求：
- 确保后续查询是自包含的，并包含网络搜索所需的必要上下文。

输出格式：
- 将您的回复格式化为具有这些确切键的JSON对象：
   - "is_sufficient": true 或 false
   - "knowledge_gap": 描述缺少什么信息或需要澄清什么
   - "follow_up_queries": 写一个具体问题来解决这个差距

示例：
{"is_sufficient": true, "knowledge_gap": "摘要缺乏性能指标和基准的信息", "follow_up_queries": ["用于评估[特定技术]的典型性能基准和指标是什么？"]}

仔细反思摘要以识别知识差距并产生后续查询。然后，按照此JSON格式生成您的输出：

摘要：
{summaries}
```

**占位符填充**：
| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{research_topic}` | `state["user_input"]` | 用户原始问题 |
| `{summaries}` | `state["web_research_result"]` 中所有摘要用 `"\n\n---\n\n"` 拼接 | 多段 web_research 摘要 |

**输出**：`is_sufficient`、`knowledge_gap`、`follow_up_queries` 写入 state，`research_loop_count += 1`

**后续路由**（`evaluate_research` 条件边，纯逻辑，不调用 LLM）：
- `is_sufficient == True` 或 `research_loop_count >= DR_MAX_LOOPS`（默认 2）→ 进入 `assess_content_quality`
- 否则 → 将 `follow_up_queries` 作为新 `web_research` 调用分派（循环回 web_research_node）

---

### 6.4 assess_content_quality_node — 内容质量评估 [DR-LLM #4]

**调用方式**：`llm.with_structured_output(ContentQualityAssessment).ainvoke(formatted_prompt)` — 结构化 JSON 输出

**触发时机**：反思判定研究充分或达到最大循环数后

**温度**：0.3

**ContentQualityAssessment Pydantic 模型**：
```python
class ContentQualityAssessment(BaseModel):
    quality_score: float                   # 0.0~1.0 质量评分
    reliability_assessment: str            # 来源可靠性评估
    content_gaps: List[str]                # 内容空白列表
    improvement_suggestions: List[str]     # 改进建议
```

**Prompt 模板**（来自 `prompts.py:content_quality_instructions`）：
```
你是一名专业的内容质量评估专家，负责评估研究内容的质量和可靠性。

指令：
- 分析提供的研究内容的整体质量
- 评估信息来源的可靠性和权威性
- 识别内容中的空白或不足之处
- 提供改进建议以提高内容质量
- 给出0.0到1.0的质量评分

评估标准：
- 信息的准确性和时效性
- 来源的权威性和可信度
- 内容的完整性和深度
- 逻辑结构和表达清晰度

输出格式：
- 将您的回复格式化为具有这些确切键的JSON对象：
   - "quality_score": 0.0到1.0的数值
   - "reliability_assessment": 可靠性评估描述
   - "content_gaps": 内容空白列表
   - "improvement_suggestions": 改进建议列表

研究主题：{research_topic}

待评估内容：
{content}
```

**占位符填充**：
| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{research_topic}` | `state["user_input"]` | 用户原始问题 |
| `{content}` | `state["web_research_result"]` 用 `"\n\n---\n\n"` 拼接 | 全部研究摘要 |

**输出**：`state["content_quality"]` 字典，包含 `quality_score`、`reliability_assessment`、`content_gaps`、`improvement_suggestions`

---

### 6.5 verify_facts_node — 事实验证 [DR-LLM #5]

**调用方式**：`llm.with_structured_output(FactVerification).ainvoke(formatted_prompt)` — 结构化 JSON 输出

**触发时机**：`assess_content_quality` 之后

**温度**：0.1（极低，强调事实准确性）

**FactVerification Pydantic 模型**：
```python
class FactVerification(BaseModel):
    verified_facts: List[Dict[str, str]]              # [{fact, source}]
    disputed_claims: List[Dict[str, str]]             # [{claim, reason}]
    verification_sources: List[Union[str, Dict]]       # 验证来源
    confidence_score: float                           # 0.0~1.0
```

**Prompt 模板**（来自 `prompts.py:fact_verification_instructions`）：
```
你是一名专业的事实核查专家，负责验证研究内容中的事实和声明。

指令：
- 识别内容中的关键事实和声明
- 验证这些事实的准确性
- 标记有争议或无法验证的声明
- 提供验证来源和置信度评分
- 当前日期是 {current_date}

验证标准：
- 事实的可验证性
- 来源的权威性
- 信息的时效性
- 数据的准确性

输出格式：
- 将您的回复格式化为具有这些确切键的JSON对象：
   - "verified_facts": 已验证事实列表，每个包含"fact"和"source"键
   - "disputed_claims": 有争议声明列表，每个包含"claim"和"reason"键
   - "verification_sources": 验证来源列表
   - "confidence_score": 0.0到1.0的置信度评分

研究主题：{research_topic}

待验证内容：
{content}
```

**占位符填充**：
| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{current_date}` | `get_current_date()` | `2026年05月14日` |
| `{research_topic}` | `state["user_input"]` | 用户原始问题 |
| `{content}` | `state["web_research_result"]` 用 `"\n\n---\n\n"` 拼接 | 全部研究摘要 |

**输出**：`state["fact_verification"]` 字典，包含 `verified_facts`、`disputed_claims`、`verification_sources`（标准化为字符串列表）、`confidence_score`

---

### 6.6 assess_relevance_node — 相关性评估 [DR-LLM #6]

**调用方式**：`llm.with_structured_output(RelevanceAssessment).ainvoke(formatted_prompt)` — 结构化 JSON 输出

**触发时机**：`verify_facts` 之后

**温度**：0.2

**RelevanceAssessment Pydantic 模型**：
```python
class RelevanceAssessment(BaseModel):
    relevance_score: float         # 0.0~1.0 相关性评分
    key_topics_covered: List[str]  # 已充分覆盖的关键主题
    missing_topics: List[str]      # 缺失或不足的主题
    content_alignment: str         # 内容与目标一致性描述
```

**Prompt 模板**（来自 `prompts.py:relevance_assessment_instructions`）：
```
你是一名专业的内容相关性分析师，负责评估研究内容与主题的相关性。

指令：
- 分析内容与研究主题的相关程度
- 识别已充分覆盖的关键主题
- 找出缺失或覆盖不足的重要主题
- 评估内容与研究目标的一致性
- 给出0.0到1.0的相关性评分

评估维度：
- 主题匹配度
- 内容深度
- 覆盖广度
- 目标一致性

输出格式：
- 将您的回复格式化为具有这些确切键的JSON对象：
   - "relevance_score": 0.0到1.0的相关性评分
   - "key_topics_covered": 已充分覆盖的关键主题列表
   - "missing_topics": 缺失或不足的主题列表
   - "content_alignment": 内容与目标一致性的描述

研究主题：{research_topic}

待评估内容：
{content}
```

**占位符填充**：
| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{research_topic}` | `state["user_input"]` | 用户原始问题 |
| `{content}` | `state["web_research_result"]` 用 `"\n\n---\n\n"` 拼接 | 全部研究摘要 |

**输出**：`state["relevance_assessment"]` 字典，包含 `relevance_score`、`key_topics_covered`、`missing_topics`、`content_alignment`

---

### 6.7 optimize_summary_node — 摘要优化 [DR-LLM #7]

**调用方式**：`llm.with_structured_output(SummaryOptimization).ainvoke(formatted_prompt)` — 结构化 JSON 输出

**触发时机**：`assess_relevance` 之后

**温度**：0.3

**SummaryOptimization Pydantic 模型**：
```python
class SummaryOptimization(BaseModel):
    optimized_summary: str       # 优化后的摘要
    key_insights: List[str]      # 关键洞察
    actionable_items: List[str]  # 可行建议
    confidence_level: str        # 置信度等级（高/中/低）
```

**Prompt 模板**（来自 `prompts.py:summary_optimization_instructions`）：
```
你是一名专业的内容优化专家，负责优化和增强研究摘要。

指令：
- 基于质量评估、事实验证和相关性分析结果优化摘要
- 提取关键洞察和发现
- 生成可行的建议和行动项
- 评估优化后内容的置信度
- 确保摘要结构清晰、逻辑严密
- 当前日期是 {current_date}

优化原则：
- 准确性优先
- 逻辑清晰
- 重点突出
- 实用性强

输出格式：
- 将您的回复格式化为具有这些确切键的JSON对象：
   - "optimized_summary": 优化后的摘要
   - "key_insights": 关键洞察列表
   - "actionable_items": 可行建议列表
   - "confidence_level": 置信度等级（高/中/低）

研究主题：{research_topic}

原始摘要：
{original_summary}

质量评估结果：
{quality_assessment}

事实验证结果：
{fact_verification}

相关性评估结果：
{relevance_assessment}
```

**占位符填充**：
| 占位符 | 来源 | 示例值 |
|--------|------|--------|
| `{current_date}` | `get_current_date()` | `2026年05月14日` |
| `{research_topic}` | `state["user_input"]` | 用户原始问题 |
| `{original_summary}` | `state["web_research_result"]` 用 `"\n\n---\n\n"` 拼接 | 全部研究摘要 |
| `{quality_assessment}` | `str(state["content_quality"])` | 质量评估字典全文 |
| `{fact_verification}` | `str(state["fact_verification"])` | 事实验证字典全文 |
| `{relevance_assessment}` | `str(state["relevance_assessment"])` | 相关性评估字典全文 |

**输出**：
- `state["summary_optimization"]`：优化结果字典
- `state["quality_enhanced_summary"]`：复制 `optimized_summary`（供 `finalize_answer` 使用）
- `state["final_confidence_score"]`：三项分数的平均值 `(quality_score + fact_confidence + relevance_score) / 3`

---

### 6.8 非 LLM 节点

以下两个节点不调用 LLM，仅做文本组合和文件写入：

#### generate_verification_report_node

将 `content_quality`、`fact_verification`、`relevance_assessment`、`summary_optimization` 四个字典格式化为一个 Markdown 验证报告字符串，存入 `state["verification_report"]`。

报告结构：内容质量评估 → 事实验证结果 → 相关性评估 → 摘要优化结果 → 综合评估（含最终置信度评分）

#### finalize_answer_node

1. 将 `quality_enhanced_summary`（优化摘要）与 `verification_report`（验证报告）拼接
2. 将短引用标记（`[1]`、`[2]`）恢复为完整 URL
3. 追加质量指标（最终置信度、内容质量评分、事实置信度、相关性评分）
4. 持久化到 `workspace/reports/{timestamp}_{thread_id}_{slug}.md`
5. 写入审计日志
6. 输出 `state["final_answer"]`

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

### 多智能体模式

| 场景 | LLM 调用次数 | 说明 |
|------|-------------|------|
| 闲聊（direct_answer） | 1 | planner 直接回复，跳过 worker/aggregator |
| 单任务 + 高置信 | 2~3 | planner + worker + (judge 跳过) + aggregator 透传 |
| 单任务 + 普通 | 3~4 | planner + worker + judge + aggregator |
| 多任务并行 | 3+N+1 | planner + N 个 worker（并行）+ aggregator |
| 多阶段 + replan | 翻倍 | 上述 + 再来一轮 planner → workers → aggregator |
| 记忆提取（每轮） | +1~2 | 压缩（条件触发）+ 提取（每轮） |

### 深度研究模式

| 场景 | LLM 调用次数 | 说明 |
|------|-------------|------|
| 单轮研究（1 查询，研究充分） | 4 | generate_query + web_research + reflection + optimize_summary |
| 单轮研究（N 查询并行，研究充分） | 3 + N | generate_query(1) + N×web_research(N) + reflection(1) + optimize_summary(1) |
| 两轮研究（N 查询 + M 后续查询） | 4 + N + M | 上述 + 第二轮 reflection + M×web_research |
| 质量保证流水线 | +3 | assess_content_quality + verify_facts + assess_relevance（固定开销，研究充分后必经） |
| 记忆提取（每轮） | +1~2 | 压缩（条件触发）+ 提取（每轮） |

> **典型场景**：`generate_query` 生成 3 个查询，第 1 轮研究不充分，生成 2 个后续查询，第 2 轮研究充分 → 总 LLM 调用 = 1 + 3 + 1 + 2 + 1 + 3（质量保证）= 11 次（不含记忆提取）
