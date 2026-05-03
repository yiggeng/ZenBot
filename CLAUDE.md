# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e .

# Run CLI interface
zenbot run

# Start Web UI (port 7860)
zenbot web

# Configure model providers interactively
zenbot config

# Real-time audit log monitor
zenbot monitor
```

No test suite is currently present. If adding tests, use `pytest`.

## Architecture

ZenBot is a LangGraph-based multi-agent framework. All requests go through a single pipeline: planner decomposes into tasks and assesses confidence, workers execute in parallel, and aggregator merges results.

### Execution Path

```
START → multi_subgraph(planner → approval → workers → aggregator) → memory_manager → END
                                    ↑                              |
                                    └──── __replan__ (reflection) ─┘
```

- **planner_node**: decomposes request into 1-4 sub-tasks with dependencies and self-assessed confidence (0~1). Never creates "summarize" or "write report" sub-tasks — workers focus solely on information gathering and action execution.
- **approval_node**: human-in-the-loop interrupt; skipped when single task + high confidence (≥0.7)
- **stage_dispatch_node**: topologically sorts tasks into sequential stages; within each stage, workers run in parallel via LangGraph's `Send` API
- **aggregator_node**: merges worker results into final answer. Performs **global reflection** — if it detects the execution plan went off track or has recoverable failures, it outputs a `__replan__`-prefixed reason, which forces a jump back to planner to restructure the plan (closed-loop error recovery).

### Key Modules

| Path | Purpose |
|------|---------|
| [zenbot/core/multi_agent.py](zenbot/core/multi_agent.py) | Main LangGraph graph definition — all nodes and edges |
| [zenbot/core/context.py](zenbot/core/context.py) | `MultiState` and `WorkerState` dataclasses |
| [zenbot/core/provider.py](zenbot/core/provider.py) | LLM factory — maps provider name to LangChain chat model |
| [zenbot/core/skill_loader.py](zenbot/core/skill_loader.py) | Loads skill packages from `workspace/office/skills/` at startup |
| [zenbot/core/tools/builtins.py](zenbot/core/tools/builtins.py) | All built-in tools (web search, file I/O, scheduler, etc.) |
| [zenbot/core/tools/sandbox_tools.py](zenbot/core/tools/sandbox_tools.py) | Sandboxed shell/file execution (restricted to `office/`) |
| [zenbot/core/logger.py](zenbot/core/logger.py) | JSONL audit logging per thread |
| [entry/cli.py](entry/cli.py) | Typer CLI entry points |
| [entry/webui.py](entry/webui.py) | Gradio Web UI (port 7860) |

### State

`MainState` is persisted in `workspace/state.sqlite3` via LangGraph's SQLite checkpointer. Key fields:

- `messages` — full conversation history (`add_messages` reducer)
- `summary` — sliding-window compressed context (triggers at ≥40 turns, keeps last 10)
- `tasks` / `stages` / `current_stage` — planner decomposition state
- `worker_results` — accumulated outputs across all workers
- `final_answer` — response returned to user

`WorkerState` is ephemeral per worker; `prev_results` is injected via `Send` so each worker sees prior-stage outputs.

### Sessions

- Default thread: `ZenBot_main` (continuous history across restarts)
- New session: `ZenBot_main_1`, `ZenBot_main_2`, ... (isolated SQLite checkpoint + log file)
- Deleting a session removes both the SQLite checkpoint and `workspace/logs/{thread_id}.jsonl`

### Dynamic Skills

Skills live in `workspace/office/skills/<skill-name>/` and contain a `SKILL.md` (instructions) plus an optional Python script. They are auto-loaded at startup and injected into the system prompt. The agent can only read/write files and execute shell commands inside `workspace/office/`.

### Configuration

Model provider credentials go in `.env`:
```
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=Qwen/Qwen2.5-72B-Instruct
OPENAI_API_KEY=...
OPENAI_API_BASE=...
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
```

Supported providers: `openai`, `anthropic`, `aliyun`, `tencent`, `z.ai`, `ollama`, and any OpenAI-compatible endpoint.
