"""
learning/strategy_selector.py — Adaptive Strategy Selection

THE LEARNING PART:
Instead of always trying strategies in a fixed order,
this module learns from past healing events and picks
the most effective strategy first.

HOW IT WORKS:
1. Look at past healing events
2. Rank strategies by success_rate × avg_improvement
3. Return strategies ordered best-first
4. Fall back to default order if no data yet

OVER TIME:
- After 0 queries: uses default order (expansion → multi → keyword → broader → refine)
- After 50 queries: reorders based on what's been working
- After 200 queries: highly optimized — skips strategies that never work for your data
"""

from app.learning.failure_logger import get_strategy_stats
from app.pipeline.strategies.base import HealingStrategy


# Default order (before we have data)
DEFAULT_ORDER = [
    "query_expansion",      # Cheapest, works for most vague queries
    "multi_query",          # Medium cost, good for ambiguous questions
    "keyword_fallback",     # Cheap, good for exact-term misses
    "broader_retrieval",    # Cheap, good for incomplete answers
    "chunk_refinement",     # Expensive, last resort
]


def get_best_strategy_order(question: str, strategies: list[HealingStrategy]) -> list[HealingStrategy]:
    """
    Return strategies ordered by likelihood of success.

    Uses historical data to determine the best order.
    Falls back to DEFAULT_ORDER if insufficient data.

    Parameters:
    -----------
    question : str
        The current question (future: use for query-type classification)
    strategies : list[HealingStrategy]
        All available strategies

    Returns:
    --------
    list[HealingStrategy] ordered best-first
    """
    stats = get_strategy_stats()

    # If we don't have enough data, use default order
    total_attempts = sum(s.get("total_attempts", 0) for s in stats)
    if total_attempts < 10:
        return _order_by_default(strategies)

    # Score each strategy: success_rate * 0.7 + avg_improvement * 0.3
    # This balances "how often it works" with "how much it improves"
    strategy_scores = {}
    for stat in stats:
        name = stat["strategy_name"]
        success_rate = stat.get("success_rate", 0)
        avg_improvement = stat.get("avg_improvement", 0)
        score = (success_rate * 0.7) + (max(0, avg_improvement) * 0.3)
        strategy_scores[name] = score

    # Sort strategies by score (highest first)
    def sort_key(strategy: HealingStrategy) -> float:
        return strategy_scores.get(strategy.name, 0.0)

    sorted_strategies = sorted(strategies, key=sort_key, reverse=True)

    return sorted_strategies


def _order_by_default(strategies: list[HealingStrategy]) -> list[HealingStrategy]:
    """Order strategies by the default priority."""
    name_to_strategy = {s.name: s for s in strategies}
    ordered = []

    for name in DEFAULT_ORDER:
        if name in name_to_strategy:
            ordered.append(name_to_strategy[name])

    # Add any strategies not in the default order at the end
    for s in strategies:
        if s not in ordered:
            ordered.append(s)

    return ordered
