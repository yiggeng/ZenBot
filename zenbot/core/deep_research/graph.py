"""
Deep Research subgraph for ZenBot.
Ported from DeepResearch-fullstack with adaptations for ZenBot's provider-agnostic LLM system.
"""
import asyncio
import json
import os
import re
from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from .state import DeepResearchState, WebSearchState
from .schemas import (
    SearchQueryList,
    Reflection,
    ContentQualityAssessment,
    FactVerification,
    RelevanceAssessment,
    SummaryOptimization,
)
from .prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    content_quality_instructions,
    fact_verification_instructions,
    relevance_assessment_instructions,
    summary_optimization_instructions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(temperature: float = 0.0):
    """Create an LLM instance using the system's configured provider."""
    from zenbot.core.provider import get_provider
    provider = os.getenv("DEFAULT_PROVIDER", "aliyun")
    model = os.getenv("DEFAULT_MODEL", "qwen-plus")
    return get_provider(provider_name=provider, model_name=model, temperature=temperature)


def _get_tavily_client():
    """Get a Tavily client using the configured API key."""
    from tavily import TavilyClient
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not configured in .env")
    return TavilyClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def generate_query_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Generate search queries based on the user's question."""
    num_queries = state.get("initial_search_query_count") or int(
        os.getenv("DR_INITIAL_QUERIES", "3")
    )

    llm = _make_llm(temperature=1.0)
    structured_llm = llm.with_structured_output(SearchQueryList)

    formatted_prompt = query_writer_instructions.format(
        current_date=get_current_date(),
        research_topic=state["user_input"],
        number_queries=num_queries,
    )
    result = await structured_llm.ainvoke(formatted_prompt)

    from zenbot.core.logger import audit_logger
    thread_id = config.get("configurable", {}).get("thread_id", "deep_research")
    audit_logger.log_event(
        thread_id=thread_id,
        event="ai_message",
        content=f"[DeepResearch] 生成 {len(result.query)} 条查询：{result.query}",
    )

    return {
        "search_query": result.query,
        "generated_queries": result.query,
    }


def _dispatch_queries(state: DeepResearchState):
    """Dispatch generated queries to parallel web_research nodes."""
    queries = state.get("search_query", [])
    return [
        Send("web_research", {"search_query": q, "id": idx})
        for idx, q in enumerate(queries)
    ]


async def web_research_node(state: WebSearchState, config: RunnableConfig) -> dict:
    """Perform web search via Tavily and summarize results with the LLM."""
    search_query = state["search_query"]
    if isinstance(search_query, list):
        search_query = search_query[0] if search_query else ""

    # --- Tavily search (sync → async) ---
    tavily_client = _get_tavily_client()
    search_results = await asyncio.to_thread(
        tavily_client.search,
        query=search_query,
        max_results=5,
        search_depth="advanced",
    )

    # --- Process results ---
    if isinstance(search_results, list):
        results_to_process = search_results
    elif isinstance(search_results, dict):
        results_to_process = search_results.get("results", [])
    elif isinstance(search_results, str):
        try:
            parsed = json.loads(search_results)
            results_to_process = parsed.get("results", []) if isinstance(parsed, dict) else [{"title": "Result", "url": "", "content": search_results}]
        except (json.JSONDecodeError, TypeError):
            results_to_process = [{"title": "Result", "url": "", "content": search_results}]
    else:
        results_to_process = []

    search_content = ""
    sources_gathered = []
    for i, result in enumerate(results_to_process):
        if isinstance(result, dict):
            title = result.get("title", f"Result {i+1}")
            url = result.get("url", "")
            content = result.get("content", str(result))
        else:
            title, url, content = f"Result {i+1}", "", str(result)

        search_content += f"Source {i+1}: {title}\nURL: {url}\nContent: {content}\n\n"
        sources_gathered.append({
            "title": title,
            "url": url,
            "content": content[:500] + "..." if len(content) > 500 else content,
            "short_url": f"[{i+1}]",
            "value": url,
            "label": title,
        })

    # --- LLM analysis ---
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=search_query,
    )
    analysis_prompt = (
        f"{formatted_prompt}\n\n搜索结果：\n{search_content}\n\n"
        "请分析这些搜索结果并提供带有引用的综合摘要。请用中文回答。"
    )

    llm = _make_llm(temperature=0.0)
    response = await llm.ainvoke(analysis_prompt)

    # Insert citation markers
    modified_text = response.content
    for source in sources_gathered:
        if source["url"] and source["url"] in modified_text:
            modified_text = modified_text.replace(source["url"], source["short_url"])
        elif source["url"]:
            domain = source["url"].split("/")[2] if len(source["url"].split("/")) > 2 else source["url"]
            if domain in modified_text:
                modified_text = modified_text.replace(domain, source["short_url"])

    return {
        "sources_gathered": sources_gathered,
        "web_research_result": [modified_text],
    }


async def reflection_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Analyze gathered research and identify knowledge gaps."""
    loop_count = state.get("research_loop_count", 0) + 1

    formatted_prompt = reflection_instructions.format(
        current_date=get_current_date(),
        research_topic=state["user_input"],
        summaries="\n\n---\n\n".join(state.get("web_research_result", [])),
    )

    llm = _make_llm(temperature=1.0)
    result = await llm.with_structured_output(Reflection).ainvoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": loop_count,
        "number_of_ran_queries": len(state.get("search_query", [])),
    }


def evaluate_research(state: DeepResearchState, config: RunnableConfig):
    """Route: continue research or proceed to quality assessment."""
    max_loops = state.get("max_research_loops") or int(
        os.getenv("DR_MAX_LOOPS", "2")
    )
    if state.get("is_sufficient") or state.get("research_loop_count", 0) >= max_loops:
        return "assess_content_quality"
    else:
        return [
            Send("web_research", {
                "search_query": q,
                "id": state.get("number_of_ran_queries", 0) + idx,
            })
            for idx, q in enumerate(state.get("follow_up_queries", []))
        ]


async def assess_content_quality_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Assess the quality and reliability of gathered research."""
    combined = "\n\n---\n\n".join(state.get("web_research_result", []))
    formatted_prompt = content_quality_instructions.format(
        research_topic=state["user_input"],
        content=combined,
    )

    llm = _make_llm(temperature=0.3)
    result = await llm.with_structured_output(ContentQualityAssessment).ainvoke(formatted_prompt)

    return {
        "content_quality": {
            "quality_score": result.quality_score,
            "reliability_assessment": result.reliability_assessment,
            "content_gaps": result.content_gaps,
            "improvement_suggestions": result.improvement_suggestions,
        }
    }


async def verify_facts_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Verify facts and claims in the research content."""
    combined = "\n\n---\n\n".join(state.get("web_research_result", []))
    formatted_prompt = fact_verification_instructions.format(
        current_date=get_current_date(),
        research_topic=state["user_input"],
        content=combined,
    )

    llm = _make_llm(temperature=0.1)
    result = await llm.with_structured_output(FactVerification).ainvoke(formatted_prompt)

    normalized_sources = []
    for src in result.verification_sources:
        if isinstance(src, dict):
            name = src.get("name") or src.get("title") or src.get("source") or ""
            desc = src.get("description") or src.get("detail") or ""
            normalized_sources.append(f"{name} - {desc}".strip(" -") if desc else name or str(src))
        else:
            normalized_sources.append(str(src))

    return {
        "fact_verification": {
            "verified_facts": result.verified_facts,
            "disputed_claims": result.disputed_claims,
            "verification_sources": normalized_sources,
            "confidence_score": result.confidence_score,
        }
    }


async def assess_relevance_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Assess content relevance to the research topic."""
    combined = "\n\n---\n\n".join(state.get("web_research_result", []))
    formatted_prompt = relevance_assessment_instructions.format(
        research_topic=state["user_input"],
        content=combined,
    )

    llm = _make_llm(temperature=0.2)
    result = await llm.with_structured_output(RelevanceAssessment).ainvoke(formatted_prompt)

    return {
        "relevance_assessment": {
            "relevance_score": result.relevance_score,
            "key_topics_covered": result.key_topics_covered,
            "missing_topics": result.missing_topics,
            "content_alignment": result.content_alignment,
        }
    }


async def optimize_summary_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Optimize and enhance the research summary."""
    original_summary = "\n\n---\n\n".join(state.get("web_research_result", []))
    formatted_prompt = summary_optimization_instructions.format(
        current_date=get_current_date(),
        research_topic=state["user_input"],
        original_summary=original_summary,
        quality_assessment=str(state.get("content_quality", {})),
        fact_verification=str(state.get("fact_verification", {})),
        relevance_assessment=str(state.get("relevance_assessment", {})),
    )

    llm = _make_llm(temperature=0.3)
    result = await llm.with_structured_output(SummaryOptimization).ainvoke(formatted_prompt)

    quality_score = state.get("content_quality", {}).get("quality_score", 0.5)
    fact_confidence = state.get("fact_verification", {}).get("confidence_score", 0.5)
    relevance_score = state.get("relevance_assessment", {}).get("relevance_score", 0.5)
    final_confidence = (quality_score + fact_confidence + relevance_score) / 3

    return {
        "summary_optimization": {
            "optimized_summary": result.optimized_summary,
            "key_insights": result.key_insights,
            "actionable_items": result.actionable_items,
            "confidence_level": result.confidence_level,
        },
        "quality_enhanced_summary": result.optimized_summary,
        "final_confidence_score": final_confidence,
    }


def generate_verification_report_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Generate a comprehensive verification report (no LLM call)."""
    quality_data = state.get("content_quality", {})
    fact_data = state.get("fact_verification", {})
    relevance_data = state.get("relevance_assessment", {})
    optimization_data = state.get("summary_optimization", {})

    report = f"""# 研究质量验证报告

## 内容质量评估
- 质量评分: {quality_data.get('quality_score', 'N/A')}/1.0
- 可靠性评估: {quality_data.get('reliability_assessment', 'N/A')}
- 内容空白: {', '.join(quality_data.get('content_gaps', []))}
- 改进建议: {', '.join(quality_data.get('improvement_suggestions', []))}

## 事实验证结果
- 验证置信度: {fact_data.get('confidence_score', 'N/A')}/1.0
- 已验证事实数量: {len(fact_data.get('verified_facts', []))}
- 争议声明数量: {len(fact_data.get('disputed_claims', []))}
- 验证来源: {', '.join(fact_data.get('verification_sources', []))}

## 相关性评估
- 相关性评分: {relevance_data.get('relevance_score', 'N/A')}/1.0
- 已覆盖关键主题: {', '.join(relevance_data.get('key_topics_covered', []))}
- 缺失主题: {', '.join(relevance_data.get('missing_topics', []))}
- 内容一致性: {relevance_data.get('content_alignment', 'N/A')}

## 摘要优化结果
- 置信度等级: {optimization_data.get('confidence_level', 'N/A')}
- 关键洞察数量: {len(optimization_data.get('key_insights', []))}
- 可行建议数量: {len(optimization_data.get('actionable_items', []))}

## 综合评估
- 最终置信度评分: {state.get('final_confidence_score', 0):.3f}/1.0
"""
    return {"verification_report": report}


def finalize_answer_node(state: DeepResearchState, config: RunnableConfig) -> dict:
    """Combine enhanced summary with verification report, restore full URLs, and persist to disk."""
    final_summary = state.get("quality_enhanced_summary") or "\n---\n\n".join(
        state.get("web_research_result", [])
    )
    verification_report = state.get("verification_report", "")

    enhanced_content = f"""{final_summary}

---

{verification_report}"""

    # Restore short citation markers to full URLs
    unique_sources = []
    for source in state.get("sources_gathered", []):
        if source["short_url"] in enhanced_content:
            enhanced_content = enhanced_content.replace(source["short_url"], source["value"])
            unique_sources.append(source)

    # Append quality metrics
    quality_metrics = "\n\n## 研究质量指标\n"
    quality_metrics += f"- 最终置信度: {state.get('final_confidence_score', 0):.3f}/1.0\n"
    quality_metrics += f"- 内容质量评分: {state.get('content_quality', {}).get('quality_score', 'N/A')}/1.0\n"
    quality_metrics += f"- 事实验证置信度: {state.get('fact_verification', {}).get('confidence_score', 'N/A')}/1.0\n"
    quality_metrics += f"- 相关性评分: {state.get('relevance_assessment', {}).get('relevance_score', 'N/A')}/1.0\n"

    final_content = enhanced_content + quality_metrics

    # Persist the report to workspace/reports/{thread}_{ts}.md
    report_path = _save_report(
        content=final_content,
        user_input=state.get("user_input", ""),
        sources=unique_sources,
        thread_id=config.get("configurable", {}).get("thread_id", "deep_research"),
    )

    if report_path:
        final_content += f"\n\n> 报告已保存至 `{report_path}`\n"

    from zenbot.core.logger import audit_logger
    thread_id = config.get("configurable", {}).get("thread_id", "deep_research")
    audit_logger.log_event(
        thread_id=thread_id,
        event="ai_message",
        content=f"[DeepResearch] 报告生成完成（{len(final_content)} 字），路径：{report_path or '未落盘'}",
    )

    return {
        "final_answer": final_content,
        "sources_gathered": unique_sources,
    }


def _save_report(content: str, user_input: str, sources: list, thread_id: str) -> str:
    """Write the deep research report to workspace/reports/ as Markdown. Returns the file path (or '' on failure)."""
    from zenbot.core.config import REPORTS_DIR

    slug = re.sub(r"[^\w\-]+", "_", (user_input or "report").strip())[:40].strip("_") or "report"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{thread_id}_{slug}.md"
    path = os.path.join(REPORTS_DIR, filename)

    header = f"# Deep Research 报告\n\n"
    header += f"- 研究主题: {user_input}\n"
    header += f"- 生成时间: {datetime.now().isoformat(timespec='seconds')}\n"
    header += f"- 会话 ID: {thread_id}\n\n---\n\n"

    body = content
    if sources:
        body += "\n\n## 参考来源\n"
        for idx, src in enumerate(sources, 1):
            title = src.get("title") or src.get("label") or f"来源 {idx}"
            url = src.get("value") or src.get("url") or ""
            body += f"{idx}. [{title}]({url})\n" if url else f"{idx}. {title}\n"

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + body)
        return path
    except OSError as e:
        print(f"[DeepResearch] 报告落盘失败: {e}")
        return ""


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def create_deep_research_subgraph():
    """Build and compile the deep research subgraph."""
    builder = StateGraph(DeepResearchState)

    # Register nodes
    builder.add_node("generate_query", generate_query_node)
    builder.add_node("web_research", web_research_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("assess_content_quality", assess_content_quality_node)
    builder.add_node("verify_facts", verify_facts_node)
    builder.add_node("assess_relevance", assess_relevance_node)
    builder.add_node("optimize_summary", optimize_summary_node)
    builder.add_node("generate_verification_report", generate_verification_report_node)
    builder.add_node("finalize_answer", finalize_answer_node)

    # Wire edges
    builder.add_edge(START, "generate_query")
    builder.add_conditional_edges("generate_query", _dispatch_queries, ["web_research"])
    builder.add_edge("web_research", "reflection")
    builder.add_conditional_edges(
        "reflection", evaluate_research, ["web_research", "assess_content_quality"]
    )
    builder.add_edge("assess_content_quality", "verify_facts")
    builder.add_edge("verify_facts", "assess_relevance")
    builder.add_edge("assess_relevance", "optimize_summary")
    builder.add_edge("optimize_summary", "generate_verification_report")
    builder.add_edge("generate_verification_report", "finalize_answer")
    builder.add_edge("finalize_answer", END)

    return builder.compile(name="deep-research")
