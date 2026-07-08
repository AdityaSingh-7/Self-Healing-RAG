"""
learning/benchmark.py — Benchmarking: Standard RAG vs Self-Healing RAG

Runs the same set of questions through BOTH pipelines and compares:
- Accuracy (confidence scores)
- Token usage
- Latency
- Which strategies helped most

This produces the numbers for your README/interview:
"Self-healing improves accuracy by X% at Y× token cost"
"""

import time
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.llm import LLMService
from app.pipeline.orchestrator import self_healing_query
from app.pipeline.validator import validate_answer
from app.learning.cost_tracker import log_cost


# Token estimation (approximate — Groq doesn't return exact counts in all cases)
# Average tokens per operation:
TOKENS_EMBED = 50          # Embedding a query
TOKENS_LLM_INPUT = 600     # Context + prompt sent to LLM
TOKENS_LLM_OUTPUT = 200    # Generated answer
TOKENS_VALIDATION = 400    # Validation prompt + response


async def run_benchmark(
    test_questions: list[dict],
    user_id: str = "benchmark_user",
) -> dict:
    """
    Run benchmark: same questions through standard and self-healing pipelines.

    Parameters:
    -----------
    test_questions : list[dict]
        Each has: {"question": str, "ground_truth": str (optional)}
    user_id : str
        Namespace to search in

    Returns:
    --------
    dict with full comparison stats
    """
    standard_results = []
    healing_results = []

    embedder = EmbeddingService()
    store = VectorStoreService()
    llm = LLMService()

    for test_case in test_questions:
        question = test_case["question"]
        ground_truth = test_case.get("ground_truth", "")

        # ==========================================
        # STANDARD PIPELINE (no healing)
        # ==========================================
        start = time.time()

        query_embedding = embedder.embed_text(question)
        results = store.search(query_embedding=query_embedding, user_id=user_id, top_k=5)

        if results:
            answer = await llm.generate(question, results)
            validation = await validate_answer(question, answer, results)
            confidence = validation["confidence"]
        else:
            answer = "No results found."
            confidence = 0.0

        standard_latency = (time.time() - start) * 1000
        standard_tokens = TOKENS_EMBED + TOKENS_LLM_INPUT + TOKENS_LLM_OUTPUT + TOKENS_VALIDATION

        # Log cost
        log_cost(
            question=question, query_type="standard", attempts=1,
            healed=False, input_tokens=TOKENS_EMBED + TOKENS_LLM_INPUT + TOKENS_VALIDATION,
            output_tokens=TOKENS_LLM_OUTPUT, latency_ms=standard_latency,
        )

        standard_results.append({
            "question": question,
            "answer": answer[:200],
            "confidence": confidence,
            "latency_ms": round(standard_latency, 1),
            "tokens": standard_tokens,
        })

        # ==========================================
        # SELF-HEALING PIPELINE
        # ==========================================
        start = time.time()

        healing_result = await self_healing_query(
            question=question, user_id=user_id, top_k=5, max_attempts=3,
        )

        healing_latency = (time.time() - start) * 1000
        attempts = healing_result["attempts"]
        # Each attempt: embed + LLM + validation. Plus strategy LLM calls.
        healing_tokens = attempts * (TOKENS_EMBED + TOKENS_LLM_INPUT + TOKENS_LLM_OUTPUT + TOKENS_VALIDATION)

        # Log cost
        log_cost(
            question=question, query_type="healing", attempts=attempts,
            healed=healing_result["healed"],
            input_tokens=int(healing_tokens * 0.7),
            output_tokens=int(healing_tokens * 0.3),
            latency_ms=healing_latency,
        )

        healing_results.append({
            "question": question,
            "answer": healing_result["answer"][:200],
            "confidence": healing_result["confidence"],
            "healed": healing_result["healed"],
            "attempts": attempts,
            "strategy_used": healing_result["strategy_used"],
            "latency_ms": round(healing_latency, 1),
            "tokens": healing_tokens,
        })

    # ==========================================
    # COMPUTE COMPARISON
    # ==========================================
    n = len(test_questions)

    std_avg_conf = sum(r["confidence"] for r in standard_results) / max(n, 1)
    heal_avg_conf = sum(r["confidence"] for r in healing_results) / max(n, 1)

    std_avg_tokens = sum(r["tokens"] for r in standard_results) / max(n, 1)
    heal_avg_tokens = sum(r["tokens"] for r in healing_results) / max(n, 1)

    std_avg_latency = sum(r["latency_ms"] for r in standard_results) / max(n, 1)
    heal_avg_latency = sum(r["latency_ms"] for r in healing_results) / max(n, 1)

    # How many were "correct" (confidence >= 0.8)
    std_correct = sum(1 for r in standard_results if r["confidence"] >= 0.8)
    heal_correct = sum(1 for r in healing_results if r["confidence"] >= 0.8)

    # How many were healed
    healed_count = sum(1 for r in healing_results if r["healed"])

    # Strategy breakdown
    strategy_counts = {}
    for r in healing_results:
        if r["strategy_used"]:
            strategy_counts[r["strategy_used"]] = strategy_counts.get(r["strategy_used"], 0) + 1

    return {
        "num_questions": n,
        "standard_pipeline": {
            "avg_confidence": round(std_avg_conf, 4),
            "correct_answers": std_correct,
            "accuracy_pct": round(std_correct / max(n, 1) * 100, 1),
            "avg_tokens": round(std_avg_tokens),
            "avg_latency_ms": round(std_avg_latency, 1),
        },
        "healing_pipeline": {
            "avg_confidence": round(heal_avg_conf, 4),
            "correct_answers": heal_correct,
            "accuracy_pct": round(heal_correct / max(n, 1) * 100, 1),
            "avg_tokens": round(heal_avg_tokens),
            "avg_latency_ms": round(heal_avg_latency, 1),
            "queries_healed": healed_count,
            "healing_rate_pct": round(healed_count / max(n, 1) * 100, 1),
        },
        "improvement": {
            "accuracy_gain_pct": round((heal_correct - std_correct) / max(n, 1) * 100, 1),
            "confidence_gain": round(heal_avg_conf - std_avg_conf, 4),
            "token_multiplier": f"{round(heal_avg_tokens / max(std_avg_tokens, 1), 2)}×",
            "latency_multiplier": f"{round(heal_avg_latency / max(std_avg_latency, 1), 2)}×",
        },
        "strategy_breakdown": strategy_counts,
        "per_question": {
            "standard": standard_results,
            "healing": healing_results,
        },
    }
