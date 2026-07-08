"""
routers/healing_query.py — Self-Healing Query Endpoint

Replaces the standard /query/ask with a self-healing version.
This is the ONLY router change — everything else is additive.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_user_id
from app.pipeline.orchestrator import self_healing_query
from app.learning.failure_logger import get_healing_summary, get_recent_healing_events, get_strategy_stats


router = APIRouter(tags=["Self-Healing Query"])


class HealingQueryRequest(BaseModel):
    """Request for the self-healing query endpoint."""
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1, le=5, description="Max healing attempts before degradation")


@router.post("/ask")
async def healing_ask(request: HealingQueryRequest, user_id: str = Depends(get_user_id)):
    """
    Ask a question with self-healing.

    The system:
    1. Retrieves context and generates an answer
    2. Validates the answer (confidence scoring via LLM-as-judge)
    3. If confidence < 0.8: automatically retries with healing strategies
    4. Returns the best answer + full healing report

    Response includes:
    - answer: the generated answer
    - confidence: how confident the system is (0.0-1.0)
    - healed: whether healing was needed
    - strategy_used: which strategy fixed it (if healed)
    - healing_report: full details of each attempt
    """
    result = await self_healing_query(
        question=request.question,
        user_id=user_id,
        top_k=request.top_k,
        recency_weight=request.recency_weight,
        max_attempts=request.max_attempts,
    )

    return result


@router.get("/explain")
async def explain_last_healing():
    """
    Get the healing report from the most recent query.
    Useful for debugging and understanding how the system heals.
    """
    events = get_recent_healing_events(limit=1)
    if not events:
        return {"message": "No healing events yet. Ask a question first."}
    return {"last_event": events[0]}


@router.get("/strategies")
async def strategy_performance():
    """
    View performance stats for each healing strategy.
    Shows which strategies work best for your data.
    """
    stats = get_strategy_stats()
    summary = get_healing_summary()

    return {
        "overall": summary,
        "strategies": stats,
    }


@router.get("/history")
async def healing_history(limit: int = 20):
    """View recent healing events."""
    events = get_recent_healing_events(limit=limit)
    return {"events": events}


# ==========================================
# BENCHMARK & COST TRACKING
# ==========================================
from app.learning.cost_tracker import get_cost_comparison, get_cost_summary
from app.learning.benchmark import run_benchmark


class BenchmarkRequest(BaseModel):
    """Request to run a benchmark comparison."""
    test_questions: list[dict] = Field(
        min_length=1,
        max_length=100,
        description="List of {question: str, ground_truth: str (optional)}",
    )


@router.post("/benchmark")
async def run_benchmark_comparison(request: BenchmarkRequest, user_id: str = Depends(get_user_id)):
    """
    Run the same questions through BOTH pipelines and compare.

    Returns:
    - Standard RAG accuracy + tokens + latency
    - Self-Healing RAG accuracy + tokens + latency
    - Improvement metrics (the interview numbers)
    - Per-question breakdown
    """
    results = await run_benchmark(
        test_questions=request.test_questions,
        user_id=user_id,
    )
    return results


@router.get("/costs")
async def cost_analysis():
    """
    Token usage and cost comparison: standard vs healed queries.

    This answers: "How much more does healing cost, and is it worth it?"
    """
    return {
        "comparison": get_cost_comparison(),
        "summary": get_cost_summary(),
    }


@router.get("/sample-benchmark")
async def sample_benchmark_dataset():
    """
    Returns a sample benchmark dataset you can use for testing.
    Replace these with questions relevant to YOUR uploaded documents.
    """
    return {
        "description": "Sample benchmark questions. Upload documents first, then run POST /healing/benchmark with these.",
        "test_questions": [
            {"question": "What is the main topic of the document?", "ground_truth": ""},
            {"question": "What are the key steps described?", "ground_truth": ""},
            {"question": "Are there any deadlines mentioned?", "ground_truth": ""},
            {"question": "What tools or technologies are referenced?", "ground_truth": ""},
            {"question": "Who is the intended audience?", "ground_truth": ""},
            {"question": "What are the prerequisites?", "ground_truth": ""},
            {"question": "Is there a troubleshooting section?", "ground_truth": ""},
            {"question": "What happens if the token expires?", "ground_truth": ""},
            {"question": "How do I verify the setup is correct?", "ground_truth": ""},
            {"question": "What are the security considerations?", "ground_truth": ""},
        ],
    }

