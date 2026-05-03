# ZenBot

> 最后更新：2026-05-03

## 项目定位

ZenBot 是一个基于 **LangGraph** 构建的本地 AI 助手，支持命令行（CLI）和 Web UI 两种运行方式。所有请求统一走多智能体管线：planner 拆解任务并评估置信度，worker 并行执行，aggregator 汇总结果。

---

## 目录结构

```
ZenBot/
├── zenbot/core/
│   ├── config.py          # 路径常量（DB_PATH、MEMORY_DIR、OFFICE_DIR 等）
│   ├── context.py         # MainState / MultiAgentState / WorkerState 定义
│   ├── multi_agent.py     # 主图工厂：multi_subgraph → memory_manager
│   ├── provider.py        # 多模型适配（Aliyun / OpenAI / Anthropic / Ollama 等）
│   ├── skill_loader.py    # 动态加载 office/skills/ 下的技能包
│   ├── heartbeat.py       # 后台心跳（定时任务触发器）
│   ├── logger.py          # 审计日志（JSONL 异步写入，按 thread_id 分文件）
│   └── tools/
│       ├── builtins.py    # 内置工具（web_search、calculator、scheduler 等）
│       ├── sandbox_tools.py  # 沙盒文件/Shell 工具
│       └── base.py        # 工具基类（zenbot_tool 装饰器 / ZenBotBaseTool）
├── entry/
│   ├── main.py            # CLI 异步主入口
│   ├── webui.py           # Gradio Web UI：对话、会话管理、实时监控
│   ├── cli.py             # CLI 命令入口（zenbot run / web / monitor / config）
│   └── monitor.py         # 实时日志监控面板（tail JSONL + Rich 渲染）
├── workspace/
│   ├── state.sqlite3      # LangGraph checkpointer（所有会话的 messages + summary）
│   ├── memory/
│   │   └── user_profile.md    # 用户长期画像（姓名、职业、偏好等）
│   └── office/            # 沙盒工作区（agent 只能读写此目录）
│       └── skills/        # 动态技能包目录
└── logs/
    └── {thread_id}.jsonl  # 审计日志（每个会话独立文件）
```

---

## 记忆体系

ZenBot 有三层记忆机制：


| 层级     | 存储位置                            | 持久性     | 用途                                         |
| -------- | ----------------------------------- | ---------- | -------------------------------------------- |
| 短期记忆 | `workspace/state.sqlite3`           | 跨重启保留 | 对话历史、上下文摘要（滑动窗口压缩）         |
| 用户画像 | `workspace/memory/user_profile.md`  | 永久       | 用户偏好、静态信息（LLM 主动调用工具保存）   |

- **用户画像**：agent 通过 `save_user_profile` 工具更新，planner/worker 启动时自动注入 system prompt
- **对话摘要**：≥40 轮时触发压缩，保留最新 10 轮，摘要注入 prompt

---

## 状态定义

### MainState（主图持久化状态，存于 SQLite）

定义在 `zenbot/core/context.py`。


| 字段           | 类型                | Reducer        | 说明                                  |
| -------------- | ------------------- | -------------- | ------------------------------------- |
| `messages`     | `List[BaseMessage]` | `add_messages` | 完整对话历史                          |
| `summary`      | `str`               | 覆盖           | 滑动窗口压缩后的上下文摘要（≤150字） |
| `user_input`   | `str`               | 覆盖           | 本轮用户原始输入                      |
| `final_answer` | `str`               | 覆盖           | 最终回复内容                          |

### MultiAgentState（多智能体子图内部状态，不持久化）

定义在 `zenbot/core/context.py`。


| 字段             | 类型               | Reducer        | 说明                                         |
| ---------------- | ------------------ | -------------- | -------------------------------------------- |
| `user_input`     | `str`              | 覆盖           | 本轮用户请求                                 |
| `tasks`          | `List[dict]`       | 覆盖           | Planner 拆解的子任务列表（含`depends_on`）   |
| `stages`         | `List[List[dict]]` | 覆盖           | 待执行的阶段队列                             |
| `current_stage`  | `List[dict]`       | 覆盖           | 当前正在执行的阶段                           |
| `worker_results` | `List[str]`        | `operator.add` | 所有 worker 的产出（并行追加合并）           |
| `final_answer`   | `str`              | 覆盖           | 汇总回复；`"__replan__"` 触发重新规划        |
| `confidence`     | `float`            | 覆盖           | planner 输出的置信度，用于 approval 跳过判断 |

### WorkerState（Worker 子图状态，不持久化）

定义在 `zenbot/core/multi_agent.py`。


| 字段               | 类型                | Reducer        | 说明                                    |
| ------------------ | ------------------- | -------------- | --------------------------------------- |
| `task_id`          | `int`               | —             | 子任务编号                              |
| `task_desc`        | `str`               | —             | 子任务描述                              |
| `prev_results`     | `List[str]`         | —             | 上一阶段产出，由`Send()` 注入           |
| `initial_messages` | `List[BaseMessage]` | —             | 初始 prompt，重试时用于重置消息历史     |
| `retry_count`      | `int`               | 覆盖           | 已重试次数（最多 3 次）                 |
| `should_retry`     | `bool`              | 覆盖           | judge 节点设置，控制是否回到 agent 重试 |
| `tool_loop_count`  | `int`               | 覆盖           | 当前 tool loop 轮数（上限 15）          |
| `worker_messages`  | `List[BaseMessage]` | `add_messages` | worker 内部消息历史                     |
| `worker_results`   | `List[str]`         | `operator.add` | 本 worker 的输出，完成后合并到子图状态  |

---

## 图结构

### 主图（MainState）

```
START → multi_subgraph → memory_manager → END
```

- **multi_subgraph**：调用多智能体子图，结果冒泡回 MainState
- **memory_manager**：滑动窗口压缩（≥40轮触发，保留最新10轮）

### 多智能体子图（MultiAgentState）

```
START → planner → approval → stage_dispatch → workers → aggregator → END
              ↑                                              |
              └──────────── __replan__ (反思重做) ───────────┘
```

- **planner**：LLM 拆解为 1-4 个子任务（含依赖关系），并自评置信度（0~1）。对于闲聊等基础交互，可直接生成 `direct_answer` 拦截。**不会额外分配"总结"或"撰写最终报告"类子任务**，所有 worker 聚焦于信息获取和操作执行。
- **approval**：人机交互审批；单任务 + 高置信（≥0.7）时跳过
- **stage_dispatch**：拓扑排序，弹出当前阶段
- **workers**：通过 Send API 并行执行
- **aggregator**：汇总所有 worker 结果。单子任务时优先原单透传；还有后续阶段则继续流转。全部完成后，aggregator 执行**全局反思**——如果判断执行计划跑偏或存在可挽救的失败，输出 `__replan__` 前缀的反思原因，系统会携带反馈强制跳回 planner 重新制定计划，形成闭环容错。

### Worker 子图（WorkerState）

```
START → agent → tools → agent → ... → collect → judge → END
                                   (最多10轮)       ↑失败重试
```

- **agent**：LLM + tools 推理
- **tools**：执行工具调用
- **collect**：收集最终结果
- **judge**：LLM 判断成功/失败，失败重试最多 3 次

---

## 核心数据流

```
用户输入
  └─► multi_subgraph_node
        └─► planner_node
              └─ LLM 输出 JSON 任务列表（含 depends_on）和置信度（confidence）
        └─► approval_node
              ├─ 单任务 + 高置信 → 跳过，直接执行
              ├─ 多任务或低置信单任务 → interrupt 等待用户确认 (y/n/建议)
              │   ├─ n → 收集建议 → replan
              │   └─ y → 继续
              └─ tasks=[] → 直接到 aggregator
        └─► stage_dispatch_node
              └─ _build_stages() 拓扑排序，弹出当前阶段
        └─► dispatch_current_stage（Send API）
              └─ 并行创建 N 个 worker
        └─► worker 子图 × N
              └─ agent → tools* → collect → judge
        └─► aggregator_node
              ├─ 还有阶段 → stage_dispatch
              ├─ __replan__（反思失败原因）→ 回到 planner 重新规划
              └─ 全部完成 → final_answer
  └─► memory_manager_node
        └─ 滑动窗口压缩（≥40轮触发）
  └─► END
```

---

## 会话管理


| 操作            | 效果                                                            |
| --------------- | --------------------------------------------------------------- |
| 正常启动        | `thread_id = "zenbot_main"`，历史从上次断点继续                 |
| CLI`/new`       | `thread_id = "zenbot_main_1/2/3..."`，新 thread 历史为空        |
| Web UI 新会话   | `thread_id = "zenbot_main_{4位随机数}"`，同上                   |
| Web UI 切换会话 | 从 SQLite 加载对应 thread_id 的历史消息，同步刷新 Monitor 日志  |
| Web UI 删除会话 | 删除 SQLite checkpoint +`.jsonl` 日志，自动切换到下一个可用会话 |

---

## 工具体系

### 内置工具


| 工具名                  | 功能                                         |
| ----------------------- | -------------------------------------------- |
| `web_search`            | Tavily 联网搜索                              |
| `get_current_time`      | 获取系统当前时间                             |
| `calculator`            | 安全数学表达式计算（AST 白名单，无 eval）    |
| `read_office_file`      | 读取 office/ 目录下的文件                    |
| `write_office_file`     | 写入/覆盖/追加 office/ 目录下的文件          |
| `list_office_files`     | 列出 office/ 目录下的文件和文件夹            |
| `execute_office_shell`  | 在 office/ 目录下执行 Shell 命令（沙盒限制） |
| `save_user_profile`     | 更新 workspace/memory/user_profile.md        |
| `get_system_model_info` | 查询当前使用的模型提供商和型号               |
| `schedule_task`         | 设置定时提醒/闹钟（支持循环）                |
| `list_scheduled_tasks`  | 查看所有待执行的定时任务                     |
| `delete_scheduled_task` | 取消指定定时任务                             |
| `modify_scheduled_task` | 修改定时任务的时间或内容                     |

### 动态技能（Skills）

存放于 `workspace/office/skills/<skill-name>/`，每个技能包含：

- `SKILL.md`：技能说明书（agent 必须先 `mode='help'` 读取）
- 可选 Python 脚本（script 型技能）

两种类型：

- **workflow 型**：只有 SKILL.md，agent 读完步骤后直接用内置工具执行
- **script 型**：含独立脚本，通过 `execute_office_shell` 运行

---

## 审计日志与 Monitor

### 日志格式（`logs/{thread_id}.jsonl`）

每行一条 JSON：

```json
{"ts": "2026-05-02T20:04:20Z", "thread_id": "zenbot_main", "event": "tool_call", "tool": "web_search", "args": {...}}
```

### 事件类型


| event           | 触发时机                       | Monitor 显示         |
| --------------- | ------------------------------ | -------------------- |
| `system_action` | worker 重试、Planner 拆解逻辑  | 黄色单行文本         |
| `tool_call`     | worker 决定调用工具            | 紫色 Panel（含参数） |
| `tool_result`   | 工具执行完毕返回结果           | 青色 Panel（含摘要） |
| `ai_message`    | worker/aggregator 生成文字回复 | 亮紫色 Panel         |

---

## Web UI（Gradio）

入口：`entry/webui.py`，访问 `http://localhost:7860`。

### 布局

```
┌──────────────────────────────┬──────────────────────────┐
│  对话区（左侧）               │  监控区（右侧）            │
│  ┌────────────────────────┐  │  实时监控                │
│  │ Chatbot（气泡布局）      │  │  切换历史会话 Dropdown    │
│  └────────────────────────┘  │  Monitor 日志 Markdown    │
│  输入框 + 发送按钮            │  刷新日志                 │
│  新会话 / 停止 / 清空         │  删除会话                 │
└──────────────────────────────┴──────────────────────────┘
```


## 启动方式

```bash
# 安装依赖
pip install -e .

# 交互式配置模型
zenbot config

# 启动 CLI
zenbot run

# 启动 Web UI（默认 7860 端口）
zenbot web

# 启动监控面板（CLI 模式下另开终端）
zenbot monitor
```

### .env 配置示例

```
DEFAULT_PROVIDER=aliyun
DEFAULT_MODEL=qwen-plus
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
TAVILY_API_KEY=tvly-xxxxx
```

支持的 Provider：`openai`、`anthropic`、`aliyun`、`tencent`、`z.ai`、`ollama` 及任意 OpenAI 兼容端点。

### CLI 命令


| 命令           | 效果                                   |
| -------------- | -------------------------------------- |
| `/new`         | 开启新会话，历史隔离                   |
| `/exit`        | 退出程序，状态持久化                   |
| `y`            | Planner 审批阶段确认执行               |
| `n`            | Planner 审批阶段拒绝，触发改进建议收集 |
| 拒绝后输入建议 | 带入反馈重新规划（replan）             |
