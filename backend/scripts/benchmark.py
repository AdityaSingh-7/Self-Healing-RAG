"""
scripts/benchmark.py — Run the Standard vs Self-Healing Comparison

Usage:
    python scripts/benchmark.py

This is THE script that produces your resume numbers.
It runs the SAME questions through both pipelines and compares:
- Accuracy (% of answers with confidence >= 0.8)
- Token usage (total tokens per query)
- Latency (ms per query)
- Which strategies helped and how much

PREREQUISITE: Run scripts/ingest.py first to load documents.
"""

import sys
import asyncio
import json
import time
sys.path.insert(0, ".")

from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.llm import LLMService
from app.pipeline.orchestrator import advanced_query
from app.pipeline.validator import validate_answer


# Test questions — EDIT THESE to match your uploaded documents
# The more specific your questions, the more meaningful the benchmark
TEST_QUESTIONS = [
    {"question": "What is the main topic of the document?"},
    {"question": "What are the key steps described?"},
    {"question": "Are there any deadlines or dates mentioned?"},
    {"question": "What tools or technologies are referenced?"},
    {"question": "Who is the intended audience?"},
    {"question": "What are the prerequisites or requirements?"},
    {"question": "Is there a troubleshooting section?"},
    {"question": "What happens if something expires or fails?"},
    {"question": "How do I verify the setup is correct?"},
    {"question": "What are the security considerations?"},
]

# Token estimates per operation (Groq doesn't always return exact counts)
TOKENS_PER_EMBED = 50
TOKENS_PER_LLM_CALL = 800      # ~600 input + ~200 output
TOKENS_PER_VALIDATION = 400     # validation prompt + response
TOKENS_PER_STRATEGY_LLM = 300  # strategy-specific LLM call (expansion, multi-query)


async def run_standard_pipeline(question: str, user_id: str) -> dict:
    """Run question through standard RAG (no healing)."""
    start = time.time()

    embedder = EmbeddingService()
    store = VectorStoreService()
    llm = LLMService()

    query_embedding = embedder.embed_text(question)
    results = store.search(query_embedding=query_embedding, user_id=user_id, top_k=5)

    if not results:
        return {"confidence": 0.0, "tokens": TOKENS_PER_EMBED, "latency_ms": 0, "answer": "No results"}

    answer = await llm.generate(question, results)

    # Validate (so we can compare confidence fairly)
    validation = await validate_answer(question, answer, results)

    latency = (time.time() - start) * 1000
    tokens = TOKENS_PER_EMBED + TOKENS_PER_LLM_CALL + TOKENS_PER_VALIDATION

    return {
        "confidence": validation["confidence"],
        "tokens": tokens,
        "latency_ms": latency,
        "answer": answer[:150],
        "reason": validation["reason"],
    }


async def run_healing_pipeline(question: str, user_id: str) -> dict:
    """Run question through self-healing RAG."""
    start = time.time()

    result = await advanced_query(
        question=question,
        user_id=user_id,
        top_k=5,
        max_attempts=3,
    )

    latency = (time.time() - start) * 1000
    attempts = result["attempts"]

    # Token estimate: initial + (attempts-1) * (strategy_llm + embed + llm + validation)
    tokens = TOKENS_PER_EMBED + TOKENS_PER_LLM_CALL + TOKENS_PER_VALIDATION
    if attempts > 1:
        tokens += (attempts - 1) * (TOKENS_PER_STRATEGY_LLM + TOKENS_PER_EMBED + TOKENS_PER_LLM_CALL + TOKENS_PER_VALIDATION)

    return {
        "confidence": result["confidence"],
        "tokens": tokens,
        "latency_ms": latency,
        "answer": result["answer"][:150],
        "healed": result["healed"],
        "attempts": attempts,
        "strategy_used": result["strategy_used"],
    }


async def main():
    user_id = "benchmark_user"

    print("=" * 70)
    print("BENCHMARK: Standard RAG vs Self-Healing RAG")
    print("=" * 70)
    print(f"Questions: {len(TEST_QUESTIONS)}")
    print(f"User namespace: {user_id}")
    print()

    # Check if there are documents
    store = VectorStoreService()
    stats = store.index.describe_index_stats()
    ns = stats.namespaces.get(user_id)
    if not ns or ns.vector_count == 0:
        print("❌ No documents found! Run scripts/ingest.py first.")
        return

    print(f"📚 Documents in index: {ns.vector_count} vectors")
    print("\n" + "-" * 70)

    standard_results = []
    healing_results = []

    for i, test in enumerate(TEST_QUESTIONS, 1):
        q = test["question"]
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {q}")

        # Standard
        std = await run_standard_pipeline(q, user_id)
        standard_results.append(std)
        print(f"   Standard: confidence={std['confidence']:.2f}, tokens={std['tokens']}, latency={std['latency_ms']:.0f}ms")

        # Healing
        heal = await run_healing_pipeline(q, user_id)
        healing_results.append(heal)
        healed_tag = f" → HEALED by {heal['strategy_used']}" if heal["healed"] else ""
        print(f"   Healing:  confidence={heal['confidence']:.2f}, tokens={heal['tokens']}, latency={heal['latency_ms']:.0f}ms, attempts={heal['attempts']}{healed_tag}")

    # ==========================================
    # COMPUTE RESULTS
    # ==========================================
    n = len(TEST_QUESTIONS)

    std_correct = sum(1 for r in standard_results if r["confidence"] >= 0.8)
    heal_correct = sum(1 for r in healing_results if r["confidence"] >= 0.8)

    std_avg_conf = sum(r["confidence"] for r in standard_results) / n
    heal_avg_conf = sum(r["confidence"] for r in healing_results) / n

    std_avg_tokens = sum(r["tokens"] for r in standard_results) / n
    heal_avg_tokens = sum(r["tokens"] for r in healing_results) / n

    std_avg_latency = sum(r["latency_ms"] for r in standard_results) / n
    heal_avg_latency = sum(r["latency_ms"] for r in healing_results) / n

    healed_count = sum(1 for r in healing_results if r["healed"])
    token_multiplier = heal_avg_tokens / max(std_avg_tokens, 1)
    latency_multiplier = heal_avg_latency / max(std_avg_latency, 1)
    accuracy_gain = ((heal_correct - std_correct) / n) * 100

    # Strategy breakdown
    strategies_used = {}
    for r in healing_results:
        if r.get("strategy_used"):
            s = r["strategy_used"]
            strategies_used[s] = strategies_used.get(s, 0) + 1

    # ==========================================
    # PRINT RESULTS
    # ==========================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"""
┌─────────────────────────┬─────────────────┬─────────────────┐
│         Metric          │   Standard RAG  │ Self-Healing RAG │
├─────────────────────────┼─────────────────┼─────────────────┤
│ Correct answers (≥0.8)  │    {std_correct}/{n} ({std_correct/n*100:.0f}%)     │    {heal_correct}/{n} ({heal_correct/n*100:.0f}%)      │
│ Avg confidence          │      {std_avg_conf:.4f}      │      {heal_avg_conf:.4f}      │
│ Avg tokens/query        │      {std_avg_tokens:.0f}        │      {heal_avg_tokens:.0f}        │
│ Avg latency (ms)        │      {std_avg_latency:.0f}        │      {heal_avg_latency:.0f}        │
└─────────────────────────┴─────────────────┴─────────────────┘

IMPROVEMENT:
  Accuracy gain:    +{accuracy_gain:.1f}% ({std_correct}/{n} → {heal_correct}/{n})
  Confidence gain:  +{heal_avg_conf - std_avg_conf:.4f}
  Token cost:       {token_multiplier:.2f}× ({"worth it" if accuracy_gain > 10 else "marginal"})
  Latency cost:     {latency_multiplier:.2f}×
  Queries healed:   {healed_count}/{n} ({healed_count/n*100:.0f}%)

STRATEGY BREAKDOWN:""")

    if strategies_used:
        for strategy, count in sorted(strategies_used.items(), key=lambda x: -x[1]):
            print(f"  {strategy}: {count} times")
    else:
        print("  (no healing needed — all answers were confident)")

    # ==========================================
    # SAVE TO FILE
    # ==========================================
    output = {
        "num_questions": n,
        "standard": {
            "correct": std_correct,
            "accuracy_pct": round(std_correct / n * 100, 1),
            "avg_confidence": round(std_avg_conf, 4),
            "avg_tokens": round(std_avg_tokens),
            "avg_latency_ms": round(std_avg_latency),
        },
        "healing": {
            "correct": heal_correct,
            "accuracy_pct": round(heal_correct / n * 100, 1),
            "avg_confidence": round(heal_avg_conf, 4),
            "avg_tokens": round(heal_avg_tokens),
            "avg_latency_ms": round(heal_avg_latency),
            "queries_healed": healed_count,
        },
        "improvement": {
            "accuracy_gain_pct": round(accuracy_gain, 1),
            "confidence_gain": round(heal_avg_conf - std_avg_conf, 4),
            "token_multiplier": round(token_multiplier, 2),
            "latency_multiplier": round(latency_multiplier, 2),
        },
        "strategy_breakdown": strategies_used,
        "per_question": {
            "standard": standard_results,
            "healing": healing_results,
        },
    }

    output_path = "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n📊 Full results saved to: {output_path}")
    print("\n" + "=" * 70)
    print("RESUME BULLET (use the real numbers above):")
    print("=" * 70)
    print(f'  "Self-healing pipeline improved answer accuracy from {std_correct/n*100:.0f}% to {heal_correct/n*100:.0f}%')
    print(f'   (+{accuracy_gain:.0f}%) at {token_multiplier:.1f}× token cost, with {healed_count}/{n} queries')
    print(f'   automatically healed via {len(strategies_used)} strategies"')
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
