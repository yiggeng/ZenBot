from __future__ import annotations

import operator
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


class DeepResearchState(TypedDict):
    """Deep research subgraph internal state."""
    messages: Annotated[list, add_messages]
    user_input: str
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    # Reflection (from reflection_node)
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    number_of_ran_queries: int
    # Generated queries (for UI display)
    generated_queries: list
    # Quality pipeline
    content_quality: dict
    fact_verification: dict
    relevance_assessment: dict
    summary_optimization: dict
    quality_enhanced_summary: str
    verification_report: str
    final_confidence_score: float
    # Output
    final_answer: str


class WebSearchState(TypedDict):
    search_query: str
    id: str
