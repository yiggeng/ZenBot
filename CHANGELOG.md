# ZenBot 开发日志

## 2026-05-03

### Planner 优化：禁止分配"总结"类子任务
- Planner prompt 新增强制规则：绝不要额外分配"总结"或"撰写最终报告"的子任务
- 所有 Worker 聚焦于实际的信息获取和操作执行，汇总由 Aggregator 统一负责

### Aggregator 全局反思与闭环重规划（`__replan__`）
- Aggregator 全部阶段执行完毕后，执行全局反思：评估结果是否实质性完成了用户请求
- 若判断执行计划跑偏或存在可挽救的失败，输出 `__replan__` 前缀的反思原因
- 系统捕获该标记后，携带反思反馈强制跳回 Planner 重新制定计划
- 形成 **出计划 → 并行执行 → 全局检阅反省 → OK 退出 / 方向不对则重做计划** 的闭环容错机制

### 图结构升级：`aggregator_next` 动态路由
- `aggregator_next` 条件边新增 `"planner"` 路径，配合 `__replan__` 标记实现回环
- 完整路由逻辑：`__replan__` → planner（重规划）/ 有后续阶段 → stage_dispatch / 全部完成 → END

### 文档同步
- README.md 和 CLAUDE.md 更新：图结构、数据流、节点描述同步反映 `__replan__` 闭环机制
- 移除 README.md 中未实现的"任务记忆"（task_patterns.md）相关描述

---

## 2026-05-02

### 架构重构：统一 Single/Multi Agent 为单一多智能体图

#### 核心变更：废弃独立 Chat Agent 路径
- 删除 `router_node`（意图路由）和 `route_decision`（chat/multi 分流），所有请求统一走多智能体管线
- 删除 `chat_agent_node`、`chat_tools_node`、`chat_should_continue`、`chat_done_node` 等 chat 专属节点
- 闲聊由 Planner 直接通过 `direct_answer` 字段拦截，无需独立路径

#### 核心变更：状态体系重构
- `AgentState` + `MultiState` 合并为 `MainState`（主图持久化）+ `MultiAgentState`（子图临时状态）
- `MainState` 仅保留 `messages`、`summary`、`user_input`、`final_answer`，存于 SQLite
- `MultiAgentState` 包含 `tasks`、`stages`、`current_stage`、`worker_results`、`confidence` 等子图内部字段
- 删除 `route` 字段和 `_merge_worker_results` 自定义 reducer（改用 `operator.add`）
- 删除 `__RESET__` sentinel 机制

#### 核心变更：记忆管理节点后置
- 滑动窗口压缩从 `router_node` 入口移至独立的 `memory_manager_node`
- 位于 `multi_subgraph` 之后、`END` 之前，对话结束后异步压缩，不阻塞路由判断

#### 新功能：Worker 自我修复机制
- `WorkerState` 新增 `initial_messages`、`retry_count`、`should_retry`、`tool_loop_count` 字段
- 新增 `worker_collect_result` 节点：收集 worker 最终结果，处理 tool loop 截断兜底
- 新增 `worker_judge_node` 节点：LLM 判断执行是否彻底失败，失败时自动重试（最多 3 次）
- 调用过工具的 worker 跳过 judge 直接放行（大概率成功）
- Worker 子图流程：`agent → tools → ... → collect → judge → END`（失败回 agent 重试）

#### 新功能：工具调用轮数上限
- 新增 `MAX_TOOL_LOOPS = 15`，Worker 工具调用超过上限时强制输出结果
- 达到上限 -2 时追加催促消息，提醒 LLM 停止调用工具

#### 新功能：Planner 置信度与 Approval 跳过
- Planner 输出新增 `confidence` 字段（0~1 浮点数）
- 单任务 + 高置信（≥0.7）时自动跳过 approval 审批，直接执行

#### 安全加固：calculator 工具重写
- 废弃 `eval()` 实现，改用 AST 白名单解析器
- 仅支持 `+`、`-`、`*`、`/`、`//`、`%`、`**` 运算符，拒绝非常量类型和超大指数
- 修复除零错误处理

#### 入口适配
- `entry/main.py`：删除 `chat_agent`、`chat_done` 节点渲染逻辑，统一由 `aggregator` 处理
- `entry/webui.py`：`chat_done` / `aggregator` 判断统一改为 `multi_subgraph`，删除 `chat_agent` 工具调用展示
- `_format_node_event` 简化，移除 chat 相关分支

#### 清理
- 删除 `agent.py`（旧单 Agent 入口，已废弃）
- 删除 `_base_sys_prompt()` 函数（chat 路径专属 system prompt 构建）

---

## 2026-04-29（本次重构）

### 架构重构：统一单/Multi Agent 为单一图

#### 核心变更：废弃独立单 Agent 模式
- 删除 `create_agent_app` 的调用入口，`agent.py` 保留但不再作为主流程
- Multi-Agent 图的 chat 路径完全覆盖原单 Agent 功能，意图路由自动分流，无需手动切换
- 删除 `/multi` 切换命令，删除 `use_multi` 标志位

#### 核心变更：统一历史记录，废弃 shared_summary 文件
- 废弃 `workspace/memory/shared_summary.md` 跨模式共享摘要文件及全部读写逻辑
- `MultiState` 新增 `summary: str` 字段，与 `AgentState` 对齐
- `MultiState.messages` 改为 `add_messages` reducer，从每轮重建改为跨轮增量累积
- 滑动窗口压缩逻辑从 `agent.py` 迁移至 `router_node` 入口（≥ 40 轮触发，保留最近 10 轮，压缩至 ≤ 150 字）
- summary 存储在 SQLite checkpointer，按 `thread_id` 自动隔离，`/new` 新会话从空白 summary 开始
- 删除 `config.py` 中的 `MULTI_SUMMARY_PATH` 常量

#### 核心变更：单/Multi 路径共享同一 thread_id
- `config` 与 `multi_config` 合并为同一对象，`thread_id` 统一为 `ZenBot_main`
- `/new` 新建会话只需更新 `config`，两条路径自动同步
- Multi 路径完成后将 `[HumanMessage, AIMessage]` 写入 `messages`，chat 后续轮次能看到完整上下文

#### Bug 修复：chat 路径看不到用户消息
- `chat_agent_node` 首轮/tool回调判断从"历史中有无 HumanMessage"改为"最后一条是否为 ToolMessage"
- 修复 `add_messages` 增量累积后历史中已有 HumanMessage 导致每轮用户输入丢失的问题

#### 新功能：阶段间上下文通过 LangGraph 状态传递
- `WorkerState` 新增 `prev_results: List[str]` 字段
- `dispatch_current_stage` 经 `Send()` 将上一阶段 `worker_results` 注入下一阶段每个 worker
- `worker_agent_node` 首次进入时将 `prev_results` 构建为 HumanMessage 插入消息列表
- 废弃此前通过字符串拼接 system prompt 传递上下文的临时方案

#### 新功能：阶段失败提前终止
- `aggregator_node` 在还有后续阶段时，先让 LLM 判断本阶段 success/failure
- failure 时清空 `stages`、生成失败原因回复、写入 `messages`，不再继续执行后续阶段

#### Monitor 修复
- `monitor.py` 日志路径从 `local_geek_master.jsonl` 更新为 `ZenBot_main.jsonl`
- `render_event` 新增 `ai_message` 事件渲染（紫色 Panel），修复 worker 回复和 chat AI 回复在 monitor 中不显示的问题
- `worker_agent_node` 日志从统一 `system_action` 改为：有工具调用记 `system_action`，有文字回复记 `ai_message`，monitor 中可见

---

## 2026-04-29

### Multi-Agent 重大升级

#### Bug 修复：messages illegal (20015)
- `MultiState.messages` 去掉 `add_messages` reducer，改为普通 `List`，每轮对话重新构建
- `chat_agent_node` 不再从 state 读取历史消息，始终从 `SystemMessage + HumanMessage` 重建，避免 checkpointer 中残留的不完整 tool_call 消息序列传入 SiliconFlow API 导致 `20015` 报错
- `chat_tools_node` 改为返回完整消息列表（`state["messages"] + result["messages"]`），保证当次 run 内工具调用链完整

#### Bug 修复：并行 Worker 消息字段冲突
- `WorkerState.messages` 重命名为 `worker_messages`，彻底隔离 worker 内部消息与主图 `MultiState.messages`
- 修复并行 worker 同时写回同名字段导致的 `INVALID_CONCURRENT_GRAPH_UPDATE` 报错

#### Bug 修复：interrupt_val 解包失败
- 修复 `__interrupt__` 事件的 `node_data` 解包逻辑，兼容 LangGraph 不同版本（tuple/list），用 `getattr(item, "value", None) or {}` 安全取值
- `_handle_node` 加入 `rdata is None` 守卫，防止并行 worker 完成时 LangGraph 发出空节点数据导致 `AttributeError`

#### Bug 修复：选 n 后无限 replan 循环
- `approval_node` 用户确认（y）时改为返回 `{"final_answer": ""}` 显式清除 replan 标记
- 修复 n → 反馈 → replan 流程后，再次输入 y 仍被 `approval_or_next` 误判为 `__replan__` 导致无限循环的问题

#### 新功能：选 n 后收集改进建议并重新规划
- `approval_node` 选 n 后触发第二次 `interrupt`，提示用户输入改进建议
- 反馈内容拼入 `user_input` 后走回 planner 重新拆解，新方案重新进入审批流程
- main.py 执行循环改为 `while True` 结构，支持任意多轮 interrupt（plan_approval → plan_feedback → 新 plan_approval）

#### 新功能：任务依赖感知，分阶段并行执行
- Planner prompt 新增 `depends_on` 字段，要求输出任务间的依赖关系
- 新增 `_build_stages()` 拓扑排序函数，将任务按依赖层级分组
- 新增 `stage_dispatch_node`：每次弹出第一阶段存入 `current_stage`，剩余存回 `stages`
- `dispatch_current_stage` 将当前阶段任务并行分发给 workers
- `aggregator_next` 条件边：`stages` 非空时返回 `stage_dispatch` 继续下一阶段，为空时走 `END` 汇总输出
- 效果：同阶段任务并行跑，后序任务等前置阶段全部完成后再启动
- `approval_node` 展示改为按阶段分组，清晰标注哪些任务并行、哪些等待

#### 系统优化
- `MultiState` 新增 `stages`、`current_stage` 字段
- main.py inputs 初始化同步新增字段
- 审批展示中模式标签改为"并行"/"分阶段"，不再显示原始 `depends_on` 数组

---

## 2026-04-28

### 新增功能

#### 联网搜索工具 (web_search)
- 集成 Tavily 搜索引擎，通过 `.env` 中的 `TAVILY_API_KEY` 配置
- 注册为内置工具，agent 可直接调用进行实时联网搜索
- `requirements.txt` 新增 `tavily-python` 依赖

#### Skill 自创建系统 (skill-creator)
- 新增 `workspace/office/skills/skill-creator/` 技能包
- 支持两种 skill 类型：
  - **workflow 型**（优先）：只有 SKILL.md，agent 读完步骤后直接用内置工具执行，适合需要 web_search、write_office_file 等内置工具的任务
  - **script 型**：含独立 Python 脚本，通过 execute_office_shell 运行，适合纯计算任务
- 内容通过临时文件传入（`tmp_steps.txt` / `tmp_script.py`），避免命令行转义问题
- 混合场景（既要内置工具又要脚本）用 workflow 型，步骤里写明 execute_office_shell 调用

#### Multi-Agent 并行执行模式
- 新增 `ZenBot/core/multi_agent.py`，实现 Planner → 并行 Workers → Aggregator 架构
- **Planner**：分析用户输入，拆解为可并行的独立子任务（1-4 个），输出 JSON 列表
- **Workers**：通过 LangGraph `Send` API 动态分发，每个 worker 独立子图，有自己的消息历史，可多轮调用工具
- **Aggregator**：汇总所有 worker 结果，生成最终回复
- 通过 `/multi` 命令切换模式，支持随时切回单 agent 模式

#### 单/Multi 模式共享记忆
- 新增 `workspace/memory/shared_summary.md` 作为两种模式的共享摘要文件
- 单 agent 每轮回复后自动更新摘要；multi agent 的 Planner 启动时读取，Aggregator 完成后写入
- 摘要超过 800 字自动压缩到 150 字以内，防止 token 膨胀
- 两种模式切换时上下文不断档

#### 会话隔离 (/new 命令)
- 输入 `/new` 开启新会话，切换到新的 `thread_id`，历史消息完全隔离
- 旧会话记录保留在 SQLite，不会丢失

### 系统优化

#### Agent 行为约束
- 系统 prompt 新增**立即行动铁律**：禁止在调用工具前输出计划性文字，先做再说
- 系统 prompt 新增**外部技能使用铁律**：所有 skill 首次调用必须先 `mode='help'` 读说明书
- skill_loader 的 help 返回末尾加入强制执行指令，workflow 型读完步骤后必须直接调工具

#### Skill 自动感知
- agent 系统 prompt 动态注入当前已加载的 skill 列表，重启后不再"忘记"自己有哪些技能

#### Monitor 日志覆盖
- multi-agent 模式的所有日志统一写入 `local_geek_master.jsonl`，`ZenBot monitor` 可正常监控
