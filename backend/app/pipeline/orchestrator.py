"""
pipeline/orchestrator.py — Advanced Self-Healing Orchestrator (v2)

THE FULL PIPELINE:
1. Classify question (simple vs complex)
2. If complex → decompose into sub-questions
3. For each (sub-)question:
   a. Generate HyDE embedding (hypothetical document)
   b. Run multi-strategy retrieval (semantic + HyDE + keyword)
   c. Fuse results with Reciprocal Rank Fusion (RRF)
   d. Rerank with Cross-Encoder
   e. Apply Contrastive filtering (remove false positives)
4. Generate answer from final context
5. Validate with LLM-as-judge
6. CALIBRATE the confidence score
7. If low confidence → HEAL using Thompson Sampling to pick strategy
8. Log everything for learning

ADVANCED TECHNIQUES USED:
- Thompson Sampling (multi-armed bandit) for strategy selection
- Query Decomposition (multi-hop reasoning)
- HyDE (Hypothetical Document Embedding)
- Cross-Encoder reranking (two-stage retrieval)
- Contrastive retrieval (negative query filtering)
- Reciprocal Rank Fusion (combine multiple search results)
- Platt Scaling (confidence calibration)
"""

import time
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.llm import LLMService
from app.pipeline.validator import validate_answer
from app.pipeline.bandit import bandit
from app.pipeline.calibrator import calibrator
from app.pipeline.decomposer import should_decompose, decompose_question
from app.pipeline.hyde import hyde_search
from app.pipeline.contrastive import contrastive_retrieval
from app.pipeline.fusion import reciprocal_rank_fusion
from app.pipeline.reranker import RerankerService
from app.pipeline.strategies.query_expansion import QueryExpansionStrategy
from app.pipeline.strategies.multi_query import MultiQueryStrategy
from app.pipeline.strategies.keyword_fallback import KeywordFallbackStrategy
from app.pipeline.strategies.broader_retrieval import BroaderRetrievalStrategy
from app.pipeline.strategies.chunk_refinement import ChunkRefinementStrategy
from app.learning.failure_logger import log_healing_event
from app.config import settings


# All available healing strategies
ALL_STRATEGIES = [
    QueryExpansionStrategy(),
    MultiQueryStrategy(),
    KeywordFallbackStrategy(),
    BroaderRetrievalStrategy(),
    ChunkRefinementStrategy(),
]

CONFIDENCE_THRESHOLD = 0.8
DEGRADATION_THRESHOLD = 0.5


async def advanced_query(
    question: str,
    user_id: str,
    top_k: int = 5,
    recency_weight: float = 0.2,
    max_attempts: int = 3,
    history: list[dict] | None = None,
    use_hyde: bool = True,
    use_contrastive: bool = True,
    use_cross_encoder: bool = True,
    use_decomposition: bool = True,
) -> dict:
    """
    Run the full advanced self-healing RAG pipeline.

    This is the main entry point. It orchestrates ALL techniques.
    """
    start_time = time.time()
    pipeline_report = {
        "techniques_used": [],
        "decomposed": False,
        "sub_questions": [],
        "hyde_hypothetical": None,
        "contrastive_negative": None,
        "fusion_sources": 0,
        "cross_encoder_applied": False,
        "attempts": [],
        "calibration_applied": False,
    }

    # ==========================================
    # STEP 1: Query Decomposition (if complex)
    # ==========================================
    questions_to_search = [question]

    if use_decomposition:
        try:
            is_complex = await should_decompose(question)
            if is_complex:
                sub_questions = await decompose_question(question)
                questions_to_search = sub_questions
                pipeline_report["decomposed"] = True
                pipeline_report["sub_questions"] = sub_questions
                pipeline_report["techniques_used"].append("query_decomposition")
        except Exception:
            pass  # Fall back to single question

    # ==========================================
    # STEP 2: Multi-Strategy Retrieval + Fusion
    # ==========================================
    all_retrieval_lists = []
    embedder = EmbeddingService()
    store = VectorStoreService()

    for q in questions_to_search:
        # Strategy A: Standard semantic search
        query_embedding = embedder.embed_text(q)
        semantic_results = store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=top_k * 3,  # Get more for fusion
            recency_weight=recency_weight,
        )
        all_retrieval_lists.append(semantic_results)

        # Strategy B: HyDE search
        if use_hyde:
            try:
                hyde_result = await hyde_search(q, user_id, top_k=top_k * 2)
                all_retrieval_lists.append(hyde_result["results"])
                pipeline_report["hyde_hypothetical"] = hyde_result["hypothetical"]
                pipeline_report["techniques_used"].append("hyde")
            except Exception:
                pass

    # ==========================================
    # STEP 3: Reciprocal Rank Fusion
    # ==========================================
    if len(all_retrieval_lists) > 1:
        fused_results = reciprocal_rank_fusion(
            ranked_lists=all_retrieval_lists,
            top_k=top_k * 2,  # Keep more for reranking
        )
        pipeline_report["fusion_sources"] = len(all_retrieval_lists)
        pipeline_report["techniques_used"].append("reciprocal_rank_fusion")
    else:
        fused_results = all_retrieval_lists[0] if all_retrieval_lists else []

    if not fused_results:
        return _no_results_response(question)

    # ==========================================
    # STEP 4: Contrastive Filtering
    # ==========================================
    if use_contrastive and fused_results:
        try:
            query_embedding = embedder.embed_text(question)
            contrastive_result = await contrastive_retrieval(
                question=question,
                results=fused_results,
                query_embedding=query_embedding,
            )
            fused_results = contrastive_result["results"]
            pipeline_report["contrastive_negative"] = contrastive_result["negative_query"]
            pipeline_report["techniques_used"].append("contrastive_retrieval")
        except Exception:
            pass

    # ==========================================
    # STEP 5: Cross-Encoder Reranking
    # ==========================================
    if use_cross_encoder and fused_results:
        try:
            reranker = RerankerService()
            fused_results = reranker.rerank(question, fused_results, top_k=top_k)
            pipeline_report["cross_encoder_applied"] = True
            pipeline_report["techniques_used"].append("cross_encoder_reranking")
        except Exception:
            # Cross-encoder model might not be downloaded — fall back to existing ranking
            fused_results = fused_results[:top_k]

    # Take top-K after all reranking
    final_results = fused_results[:top_k]

    # ==========================================
    # STEP 6: Generate Answer
    # ==========================================
    llm = LLMService()
    answer = await llm.generate(question, final_results, history)

    # ==========================================
    # STEP 7: Validate + Calibrate
    # ==========================================
    validation = await validate_answer(question, answer, final_results)
    raw_confidence = validation["confidence"]

    # Apply Platt Scaling calibration
    calibrated_confidence = calibrator.calibrate(raw_confidence)
    pipeline_report["calibration_applied"] = calibrator._is_fitted

    if calibrator._is_fitted:
        pipeline_report["techniques_used"].append("platt_scaling_calibration")

    confidence = calibrated_confidence

    pipeline_report["attempts"].append({
        "attempt": 1,
        "strategy": "advanced_pipeline",
        "raw_confidence": raw_confidence,
        "calibrated_confidence": calibrated_confidence,
        "reason": validation["reason"],
    })

    # ==========================================
    # STEP 8: Self-Healing (if needed)
    # ==========================================
    if confidence >= CONFIDENCE_THRESHOLD:
        return _build_response(
            answer=answer, results=final_results, confidence=confidence,
            healed=False, attempts=1, strategy_used=None,
            pipeline_report=pipeline_report,
            latency_ms=(time.time() - start_time) * 1000,
        )

    # Use Thompson Sampling to select strategy order
    strategy_order = bandit.select_strategy(ALL_STRATEGIES)
    pipeline_report["techniques_used"].append("thompson_sampling")

    for attempt_num, strategy in enumerate(strategy_order[:max_attempts - 1], start=2):
        try:
            strategy_result = await strategy.execute(
                question=question,
                original_results=final_results,
                user_id=user_id,
                validation_issues=validation["issues"],
            )
        except Exception as e:
            pipeline_report["attempts"].append({
                "attempt": attempt_num,
                "strategy": strategy.name,
                "raw_confidence": 0.0,
                "reason": f"Strategy error: {str(e)}",
            })
            bandit.update(strategy.name, success=False, reward=-0.1)
            continue

        new_results = strategy_result["results"]
        if not new_results:
            bandit.update(strategy.name, success=False, reward=0.0)
            continue

        # Generate new answer
        modified_question = strategy_result.get("modified_question", question)
        new_answer = await llm.generate(modified_question, new_results, history)

        # Validate
        new_validation = await validate_answer(question, new_answer, new_results)
        new_raw_conf = new_validation["confidence"]
        new_calibrated_conf = calibrator.calibrate(new_raw_conf)

        pipeline_report["attempts"].append({
            "attempt": attempt_num,
            "strategy": strategy.name,
            "raw_confidence": new_raw_conf,
            "calibrated_confidence": new_calibrated_conf,
            "reason": new_validation["reason"],
        })

        # Did it improve?
        if new_calibrated_conf >= CONFIDENCE_THRESHOLD:
            # Success! Update bandit with positive reward
            reward = new_calibrated_conf - confidence
            bandit.update(strategy.name, success=True, reward=reward)

            log_healing_event(
                question=question, strategy_name=strategy.name,
                confidence_before=confidence, confidence_after=new_calibrated_conf,
                success=True,
            )

            return _build_response(
                answer=new_answer, results=new_results,
                confidence=new_calibrated_conf, healed=True,
                attempts=attempt_num, strategy_used=strategy.name,
                pipeline_report=pipeline_report,
                latency_ms=(time.time() - start_time) * 1000,
            )

        # Partial improvement — update bandit
        improvement = new_calibrated_conf - confidence
        bandit.update(strategy.name, success=False, reward=improvement)

        if new_calibrated_conf > confidence:
            answer = new_answer
            final_results = new_results
            confidence = new_calibrated_conf
            validation = new_validation

    # ==========================================
    # GRACEFUL DEGRADATION
    # ==========================================
    log_healing_event(
        question=question, strategy_name="all_failed",
        confidence_before=pipeline_report["attempts"][0]["calibrated_confidence"],
        confidence_after=confidence, success=False,
    )

    if confidence < DEGRADATION_THRESHOLD:
        answer = _build_degraded_answer(question, answer, validation["issues"], final_results)

    return _build_response(
        answer=answer, results=final_results, confidence=confidence,
        healed=False, attempts=len(pipeline_report["attempts"]),
        strategy_used=None, pipeline_report=pipeline_report,
        latency_ms=(time.time() - start_time) * 1000,
        degraded=confidence < DEGRADATION_THRESHOLD,
    )


def _build_response(answer, results, confidence, healed, attempts, strategy_used, pipeline_report, latency_ms, degraded=False):
    sources = [
        {
            "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
            "filename": r.get("metadata", {}).get("filename", "unknown"),
            "page": r.get("metadata", {}).get("page", 0),
            "score": round(r.get("score", r.get("final_score", 0)), 4),
            "final_score": round(r.get("reranked_score", r.get("contrastive_score", r.get("rrf_score", r.get("final_score", 0)))), 4),
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
        "pipeline_report": pipeline_report,
    }


def _build_degraded_answer(question, partial_answer, issues, results):
    response = "⚠️ **Low Confidence Answer**\n\n"
    response += f"{partial_answer}\n\n---\n"
    response += f"**What might be missing:** {issues}\n"
    response += "**Suggestion:** Try rephrasing with more specific terms, or upload additional documents.\n"
    return response


def _no_results_response(question):
    return {
        "answer": "No documents found. Please upload relevant documents first.",
        "sources": [], "confidence": 0.0, "healed": False, "degraded": True,
        "attempts": 1, "strategy_used": None, "latency_ms": 0,
        "pipeline_report": {"techniques_used": [], "note": "No documents in store"},
    }
