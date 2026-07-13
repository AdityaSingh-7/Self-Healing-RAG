"""
routers/healing_query.py — Self-Healing Query Endpoint

Replaces the standard /query/ask with a self-healing version.
This is the ONLY router change — everything else is additive.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_user_id
from app.pipeline.orchestrator import advanced_query
from app.pipeline.bandit import bandit
from app.pipeline.calibrator import calibrator
from app.learning.failure_logger import get_healing_summary, get_recent_healing_events, get_strategy_stats


router = APIRouter(tags=["Self-Healing Query"])


class HealingQueryRequest(BaseModel):
    """Request for the self-healing query endpoint."""
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1, le=5)
    use_hyde: bool = Field(default=True, description="Use Hypothetical Document Embedding")
    use_contrastive: bool = Field(default=True, description="Use Contrastive Retrieval")
    use_cross_encoder: bool = Field(default=True, description="Use Cross-Encoder Reranking")
    use_decomposition: bool = Field(default=True, description="Use Query Decomposition for complex questions")


@router.post("/ask")
async def healing_ask(request: HealingQueryRequest, user_id: str = Depends(get_user_id)):
    """
    Ask a question with the FULL advanced pipeline:

    1. Query Decomposition (multi-hop for complex questions)
    2. HyDE (Hypothetical Document Embedding)
    3. Multi-strategy retrieval + Reciprocal Rank Fusion
    4. Contrastive filtering (negative query)
    5. Cross-Encoder reranking
    6. LLM answer generation
    7. Confidence validation + Platt Scaling calibration
    8. Self-healing with Thompson Sampling strategy selection
    """
    result = await advanced_query(
        question=request.question,
        user_id=user_id,
        top_k=request.top_k,
        recency_weight=request.recency_weight,
        max_attempts=request.max_attempts,
        use_hyde=request.use_hyde,
        use_contrastive=request.use_contrastive,
        use_cross_encoder=request.use_cross_encoder,
        use_decomposition=request.use_decomposition,
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


@router.get("/bandit")
async def bandit_stats():
    """
    Thompson Sampling bandit statistics.
    Shows the Beta distribution parameters and expected values for each strategy arm.
    """
    from app.pipeline.strategies.query_expansion import QueryExpansionStrategy
    from app.pipeline.strategies.multi_query import MultiQueryStrategy
    from app.pipeline.strategies.keyword_fallback import KeywordFallbackStrategy
    from app.pipeline.strategies.broader_retrieval import BroaderRetrievalStrategy
    from app.pipeline.strategies.chunk_refinement import ChunkRefinementStrategy

    all_strategies = [
        QueryExpansionStrategy(), MultiQueryStrategy(),
        KeywordFallbackStrategy(), BroaderRetrievalStrategy(),
        ChunkRefinementStrategy(),
    ]

    return {
        "arm_stats": bandit.get_stats(),
        "expected_values": bandit.get_expected_values(all_strategies),
    }


@router.get("/calibration")
async def calibration_info():
    """
    Confidence calibration status.
    Shows whether Platt Scaling is active and how it maps raw → calibrated scores.
    """
    return calibrator.get_calibration_info()


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

