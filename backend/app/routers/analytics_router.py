"""
routers/analytics_router.py — Analytics & Feedback Endpoints

Provides:
- POST /analytics/feedback — Submit thumbs up/down on answers
- GET /analytics/summary — View system performance metrics
- GET /analytics/recent — View recent queries and their scores
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_user_id
from app.services.analytics import get_analytics_summary, get_recent_queries, log_feedback


router = APIRouter(tags=["Analytics"])


class FeedbackRequest(BaseModel):
    """User feedback on an answer."""
    question: str = Field(description="The question that was asked")
    answer: str = Field(description="The answer that was generated")
    rating: int = Field(ge=1, le=5, description="Rating: 1=terrible, 5=excellent")
    comment: str = Field(default="", description="Optional comment explaining the rating")


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest, user_id: str = Depends(get_user_id)):
    """
    Submit feedback on an answer (thumbs up/down).
    This helps track which documents and retrieval strategies work best.
    """
    log_feedback(
        user_id=user_id,
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        comment=request.comment,
    )
    return {"status": "success", "message": "Feedback recorded. Thank you!"}


@router.get("/summary")
async def analytics_summary(days: int = 7):
    """
    Get system analytics summary for the last N days.
    Shows query volume, average latency, satisfaction scores, etc.
    """
    return get_analytics_summary(days=days)


@router.get("/recent")
async def recent_queries(limit: int = 20):
    """
    Get the most recent queries with their performance metrics.
    Useful for debugging retrieval quality.
    """
    return {"queries": get_recent_queries(limit=limit)}
