"""
routers/query.py — Query Endpoint (Enhanced)

Features:
- Semantic caching (instant response for repeated questions)
- Rate limiting (30 queries/min)
- Analytics logging (latency, scores, tokens)
- Conversation memory + query rewriting
- SSE streaming
"""

import json
import time

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.query import QueryResponse, QueryWithHistoryRequest
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.llm import LLMService
from app.services.memory import conversation_memory
from app.services.rewriter import rewrite_query
from app.services.cache import semantic_cache
from app.services.rate_limiter import query_limiter
from app.services.analytics import log_query
from app.dependencies import get_user_id


router = APIRouter(tags=["Query"])


@router.post("/ask")
async def ask_question(
    request: QueryWithHistoryRequest,
    user_id: str = Depends(get_user_id),
):
    """
    Ask a question about your documents.

    Features:
    - Semantic cache (instant if same question asked before)
    - Query rewriting (for vague follow-ups)
    - Hybrid search + recency reranking
    - SSE streaming or JSON response
    - Analytics logging
    """
    start_time = time.time()

    # ==========================================
    # RATE LIMITING
    # ==========================================
    if not query_limiter.allow(user_id):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max 30 queries/min. Try again in {query_limiter.reset_time(user_id):.0f}s.",
        )

    # ==========================================
    # QUERY REWRITING
    # ==========================================
    history = [msg.model_dump() for msg in request.history]
    if not history:
        history = conversation_memory.get_history(user_id)

    effective_question = await rewrite_query(request.question, history)
    was_rewritten = effective_question != request.question

    # ==========================================
    # EMBED QUESTION
    # ==========================================
    embedder = EmbeddingService()
    query_embedding = embedder.embed_text(effective_question)

    # ==========================================
    # SEMANTIC CACHE CHECK
    # ==========================================
    cache_hit = semantic_cache.get(query_embedding)
    if cache_hit and not request.stream:
        # Log cache hit
        latency_ms = (time.time() - start_time) * 1000
        try:
            log_query(
                user_id=user_id, question=request.question,
                effective_question=effective_question,
                answer_preview=cache_hit["answer"][:200],
                top_k=request.top_k, recency_weight=request.recency_weight,
                num_results=len(cache_hit["sources"]), avg_similarity=cache_hit["cache_similarity"],
                avg_recency=0, latency_ms=latency_ms, model="cache",
                was_rewritten=was_rewritten,
            )
        except Exception:
            pass

        return {
            "answer": cache_hit["answer"],
            "sources": cache_hit["sources"],
            "query": effective_question,
            "model": "cache (semantic hit)",
            "cached": True,
            "cache_similarity": cache_hit["cache_similarity"],
            "latency_ms": round(latency_ms, 1),
        }

    # ==========================================
    # SEARCH PINECONE
    # ==========================================
    store = VectorStoreService()
    results = store.search(
        query_embedding=query_embedding,
        user_id=user_id,
        top_k=request.top_k,
        recency_weight=request.recency_weight,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Upload some documents first.",
        )

    # ==========================================
    # GENERATE ANSWER
    # ==========================================
    llm = LLMService()

    if request.stream:
        return StreamingResponse(
            _stream_response(llm, request.question, effective_question, results, history, user_id, query_embedding, start_time, was_rewritten, request.top_k, request.recency_weight),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
    else:
        answer = await llm.generate(effective_question, results, history)
        latency_ms = (time.time() - start_time) * 1000

        # Save to memory
        conversation_memory.add_message(user_id, "user", request.question)
        conversation_memory.add_message(user_id, "assistant", answer)

        # Cache the result
        sources = _format_sources(results)
        semantic_cache.put(effective_question, query_embedding, answer, sources)

        # Log analytics
        avg_sim = sum(r["score"] for r in results) / len(results) if results else 0
        avg_rec = sum(r["recency_score"] for r in results) / len(results) if results else 0
        try:
            log_query(
                user_id=user_id, question=request.question,
                effective_question=effective_question, answer_preview=answer[:200],
                top_k=request.top_k, recency_weight=request.recency_weight,
                num_results=len(results), avg_similarity=avg_sim, avg_recency=avg_rec,
                latency_ms=latency_ms, model=llm.model, was_rewritten=was_rewritten,
            )
        except Exception:
            pass

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=effective_question,
            model=llm.model,
        )


def _format_sources(results: list[dict]) -> list[dict]:
    """Format search results into source citations."""
    return [
        {
            "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
            "filename": r["metadata"].get("filename", "unknown"),
            "page": r["metadata"].get("page", 0),
            "score": round(r["score"], 4),
            "recency_score": round(r["recency_score"], 4),
            "final_score": round(r["final_score"], 4),
        }
        for r in results
    ]


async def _stream_response(
    llm, original_question, effective_question, results, history, user_id, query_embedding, start_time, was_rewritten, top_k, recency_weight
):
    """SSE streaming generator with analytics."""
    sources = _format_sources(results)
    yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

    if was_rewritten:
        yield f"data: {json.dumps({'type': 'rewritten', 'content': effective_question})}\n\n"

    full_answer = ""
    async for token in llm.generate_stream(effective_question, results, history):
        full_answer += token
        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

    # Post-stream: save memory, cache, analytics
    conversation_memory.add_message(user_id, "user", original_question)
    conversation_memory.add_message(user_id, "assistant", full_answer)
    semantic_cache.put(effective_question, query_embedding, full_answer, sources)

    latency_ms = (time.time() - start_time) * 1000
    avg_sim = sum(r["score"] for r in results) / len(results) if results else 0
    avg_rec = sum(r["recency_score"] for r in results) / len(results) if results else 0
    try:
        log_query(
            user_id=user_id, question=original_question,
            effective_question=effective_question, answer_preview=full_answer[:200],
            top_k=top_k, recency_weight=recency_weight,
            num_results=len(results), avg_similarity=avg_sim, avg_recency=avg_rec,
            latency_ms=latency_ms, model=llm.model, was_rewritten=was_rewritten,
        )
    except Exception:
        pass

    yield f"data: {json.dumps({'type': 'done', 'latency_ms': round(latency_ms, 1)})}\n\n"


@router.post("/clear-history")
async def clear_history(user_id: str = Depends(get_user_id)):
    """Clear conversation history and semantic cache."""
    conversation_memory.clear(user_id)
    semantic_cache.clear()
    return {"status": "success", "message": "Conversation history and cache cleared."}
