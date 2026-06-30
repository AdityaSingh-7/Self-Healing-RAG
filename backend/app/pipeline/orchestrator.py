"""
pipeline/orchestrator.py — Self-Healing Orchestrator

THE BRAIN: Controls the entire healing loop.

Flow:
1. Run normal retrieval + generation
2. Validate the answer (confidence scoring)
3. If confidence >= threshold → return (healthy answer)
4. If confidence < threshold → pick a healing strategy → retry
5. Max 3 attempts, then graceful degradation

STRATEGY SELECTION:
- Default order: query_expansion → multi_query → keyword_fallback → broader_retrieval → chunk_refinement
- Over time, the adaptive learner reorders based on what works
"""

import time
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.llm import LLMService
from app.pipeline.validator import validate_answer
from app.pipeline.strategies.query_expansion import QueryExpansionStrategy
from app.pipeline.strategies.multi_query import MultiQueryStrategy
from app.pipeline.strategies.keyword_fallback import KeywordFallbackStrategy
from app.pipeline.strategies.broader_retrieval import BroaderRetrievalStrategy
from app.pipeline.strategies.chunk_refinement import ChunkRefinementStrategy
from app.learning.failure_logger import log_healing_event
from app.learning.strategy_selector import get_best_strategy_order
from app.config import settings


# All available strategies
ALL_STRATEGIES = [
    QueryExpansionStrategy(),
    MultiQueryStrategy(),
    KeywordFallbackStrategy(),
    BroaderRetrievalStrategy(),
    ChunkRefinementStrategy(),
]

# Thresholds
CONFIDENCE_THRESHOLD = 0.8   # Above this = healthy, return answer
DEGRADATION_THRESHOLD = 0.5  # Below this after all retries = admit uncertainty


async def self_healing_query(
    question: str,
    user_id: str,
    top_k: int = 5,
    recency_weight: float = 0.2,
    max_attempts: int = 3,
    history: list[dict] | None = None,
) -> dict:
    """
    Run the self-healing RAG pipeline.

    Returns:
    --------
    dict with:
        - answer: str
        - sources: list[dict]
        - confidence: float
        - healed: bool (whether healing was needed)
        - attempts: int (how many tries it took)
        - strategy_used: str | None (which strategy fixed it)
        - healing_report: dict (full details for debugging)
    """
    start_time = time.time()
    healing_report = {
        "attempts": [],
        "original_confidence": None,
        "final_confidence": None,
        "healed": False,
        "strategy_used": None,
    }

    # ==========================================
    # ATTEMPT 1: Normal pipeline
    # ==========================================
    embedder = EmbeddingService()
    store = VectorStoreService()
    llm = LLMService()

    query_embedding = embedder.embed_text(question)
    results = store.search(
        query_embedding=query_embedding,
        user_id=user_id,
        top_k=top_k,
        recency_weight=recency_weight,
    )

    if not results:
        return _no_results_response(question)

    # Generate initial answer
    answer = await llm.generate(question, results, history)

    # Validate
    validation = await validate_answer(question, answer, results)
    confidence = validation["confidence"]
    healing_report["original_confidence"] = confidence
    healing_report["attempts"].append({
        "attempt": 1,
        "strategy": "none (initial)",
        "confidence": confidence,
        "reason": validation["reason"],
    })

    # ==========================================
    # CHECK: Is the answer good enough?
    # ==========================================
    if confidence >= CONFIDENCE_THRESHOLD:
        # Healthy answer — no healing needed
        return _build_response(
            answer=answer,
            results=results,
            confidence=confidence,
            healed=False,
            attempts=1,
            strategy_used=None,
            healing_report=healing_report,
            latency_ms=(time.time() - start_time) * 1000,
        )

    # ==========================================
    # HEALING LOOP: Try strategies until confident
    # ==========================================
    # Get optimal strategy order (learned from past successes)
    strategy_order = get_best_strategy_order(question, ALL_STRATEGIES)

    for attempt_num, strategy in enumerate(strategy_order[:max_attempts - 1], start=2):
        # Execute the healing strategy
        try:
            strategy_result = await strategy.execute(
                question=question,
                original_results=results,
                user_id=user_id,
                validation_issues=validation["issues"],
            )
        except Exception as e:
            healing_report["attempts"].append({
                "attempt": attempt_num,
                "strategy": strategy.name,
                "confidence": 0.0,
                "reason": f"Strategy failed: {str(e)}",
            })
            continue

        new_results = strategy_result["results"]
        if not new_results:
            continue

        # Generate new answer with healed results
        modified_question = strategy_result.get("modified_question", question)
        new_answer = await llm.generate(modified_question, new_results, history)

        # Re-validate
        new_validation = await validate_answer(question, new_answer, new_results)
        new_confidence = new_validation["confidence"]

        healing_report["attempts"].append({
            "attempt": attempt_num,
            "strategy": strategy.name,
            "confidence": new_confidence,
            "reason": new_validation["reason"],
            "metadata": strategy_result.get("metadata", {}),
        })

        # Did it improve?
        if new_confidence >= CONFIDENCE_THRESHOLD:
            # Healed successfully!
            healing_report["healed"] = True
            healing_report["final_confidence"] = new_confidence
            healing_report["strategy_used"] = strategy.name

            # Log the healing event for learning
            log_healing_event(
                question=question,
                strategy_name=strategy.name,
                confidence_before=confidence,
                confidence_after=new_confidence,
                success=True,
            )

            return _build_response(
                answer=new_answer,
                results=new_results,
                confidence=new_confidence,
                healed=True,
                attempts=attempt_num,
                strategy_used=strategy.name,
                healing_report=healing_report,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Update for next iteration if this was better
        if new_confidence > confidence:
            answer = new_answer
            results = new_results
            confidence = new_confidence
            validation = new_validation

    # ==========================================
    # GRACEFUL DEGRADATION: All strategies exhausted
    # ==========================================
    healing_report["final_confidence"] = confidence

    # Log the failure for learning
    log_healing_event(
        question=question,
        strategy_name="all_failed",
        confidence_before=healing_report["original_confidence"],
        confidence_after=confidence,
        success=False,
    )

    # If confidence is still very low, admit uncertainty
    if confidence < DEGRADATION_THRESHOLD:
        answer = _build_degraded_answer(question, answer, validation["issues"], results)

    return _build_response(
        answer=answer,
        results=results,
        confidence=confidence,
        healed=False,
        attempts=len(healing_report["attempts"]),
        strategy_used=None,
        healing_report=healing_report,
        latency_ms=(time.time() - start_time) * 1000,
        degraded=confidence < DEGRADATION_THRESHOLD,
    )


def _build_response(
    answer: str,
    results: list[dict],
    confidence: float,
    healed: bool,
    attempts: int,
    strategy_used: str | None,
    healing_report: dict,
    latency_ms: float,
    degraded: bool = False,
) -> dict:
    """Build the standard response format."""
    sources = [
        {
            "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
            "filename": r["metadata"].get("filename", "unknown"),
            "page": r["metadata"].get("page", 0),
            "score": round(r["score"], 4),
            "final_score": round(r["final_score"], 4),
        }
        for r in results[:5]
    ]

    return {
        "answer": answer,
        "sources": sources,
        "confidence": round(confidence, 4),
        "healed": healed,
        "degraded": degraded,
        "attempts": attempts,
        "strategy_used": strategy_used,
        "latency_ms": round(latency_ms, 1),
        "healing_report": healing_report,
    }


def _build_degraded_answer(question: str, partial_answer: str, issues: str, results: list[dict]) -> str:
    """Build a graceful degradation response."""
    response = "⚠️ **Low Confidence Answer**\n\n"
    response += "I found some information but I'm not fully confident in this answer:\n\n"
    response += f"{partial_answer}\n\n"
    response += "---\n"
    response += f"**What might be missing:** {issues}\n\n"

    if results:
        response += "**Documents searched:**\n"
        filenames = set(r["metadata"].get("filename", "?") for r in results[:5])
        for fn in filenames:
            response += f"- {fn}\n"

    response += "\n💡 **Suggestions:** Try uploading more relevant documents, or rephrase your question with more specific terms."
    return response


def _no_results_response(question: str) -> dict:
    """Response when no documents are found at all."""
    return {
        "answer": "No documents found. Please upload relevant documents first, then try again.",
        "sources": [],
        "confidence": 0.0,
        "healed": False,
        "degraded": True,
        "attempts": 1,
        "strategy_used": None,
        "latency_ms": 0,
        "healing_report": {"attempts": [], "note": "No documents in vector store"},
    }
