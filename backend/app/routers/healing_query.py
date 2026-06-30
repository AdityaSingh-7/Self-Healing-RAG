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
