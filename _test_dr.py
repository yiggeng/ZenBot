import asyncio
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv
load_dotenv()

from zenbot.core.deep_research.graph import create_deep_research_subgraph

async def test():
    g = create_deep_research_subgraph()
    sub_input = {
        "messages": [],
        "user_input": "what to eat for dinner to lose weight",
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": 2,
        "max_research_loops": 1,
        "research_loop_count": 0,
        "generated_queries": [],
        "content_quality": {},
        "fact_verification": {},
        "relevance_assessment": {},
        "summary_optimization": {},
        "quality_enhanced_summary": "",
        "verification_report": "",
        "final_confidence_score": 0.0,
        "final_answer": "",
    }
    print("Starting deep research subgraph...")
    try:
        result = await g.ainvoke(sub_input, {"configurable": {"thread_id": "test_dr"}})
        print("Result keys:", list(result.keys()))
        fa = result.get("final_answer", "")
        print(f"final_answer length: {len(fa)}")
        if fa:
            print("HAS ANSWER")
        else:
            print("EMPTY ANSWER!")
            for k, v in result.items():
                if isinstance(v, str):
                    print(f"  {k}: str len={len(v)}")
                elif isinstance(v, list):
                    print(f"  {k}: list len={len(v)}")
                elif isinstance(v, dict):
                    print(f"  {k}: dict")
                else:
                    print(f"  {k}: {type(v).__name__} = {v}")
    except Exception as e:
        print(f"EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

asyncio.run(test())
