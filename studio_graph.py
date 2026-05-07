"""
LangGraph Studio 入口模块
暴露编译好的 graph 供 Studio 可视化调试
"""
import os
from dotenv import load_dotenv

load_dotenv()

from zenbot.core.multi_agent import create_multi_agent_app

provider = os.getenv("DEFAULT_PROVIDER", "openai")
model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

graph = create_multi_agent_app(
    provider_name=provider,
    model_name=model,
    checkpointer=None,  # Studio 用 MemorySaver，不影响主应用的 SQLite
)
