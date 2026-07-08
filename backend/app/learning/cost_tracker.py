"""
learning/cost_tracker.py — Token & Cost Tracking

Tracks token usage per query to measure the cost of healing.
Answers the question: "How much more does healing cost vs standard RAG?"

WHAT IT TRACKS:
- Tokens per query (embed calls, LLM calls, validation calls)
- Cost per query (based on Groq pricing)
- Comparison: standard vs healed queries
- Running averages over time

GROQ PRICING (as of 2024):
- Llama 3.3 70B: $0.59/M input tokens, $0.79/M output tokens
- We estimate ~800 tokens for a standard query, ~2400 for a healed query
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "healing.db"

# Groq Llama 3.3 70B pricing (per token)
INPUT_COST_PER_TOKEN = 0.59 / 1_000_000   # $0.59 per million input tokens
OUTPUT_COST_PER_TOKEN = 0.79 / 1_000_000   # $0.79 per million output tokens


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_cost_db():
    """Create cost tracking table."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cost_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            query_type TEXT NOT NULL,
            attempts INTEGER NOT NULL,
            healed BOOLEAN NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            estimated_cost_usd REAL NOT NULL,
            latency_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_cost(
    question: str,
    query_type: str,
    attempts: int,
    healed: bool,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
):
    """Log token usage and cost for a query."""
    total_tokens = input_tokens + output_tokens
    cost = (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)

    conn = _get_connection()
    conn.execute(
        """INSERT INTO cost_logs
        (timestamp, question, query_type, attempts, healed, input_tokens, output_tokens, total_tokens, estimated_cost_usd, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            question, query_type, attempts, healed,
            input_tokens, output_tokens, total_tokens,
            round(cost, 8), latency_ms,
        ),
    )
    conn.commit()
    conn.close()


def get_cost_comparison() -> dict:
    """
    Compare costs: standard queries vs healed queries.
    This is the money slide for interviews.
    """
    conn = _get_connection()

    standard = conn.execute("""
        SELECT
            COUNT(*) as count,
            AVG(total_tokens) as avg_tokens,
            AVG(estimated_cost_usd) as avg_cost,
            AVG(latency_ms) as avg_latency,
            SUM(estimated_cost_usd) as total_cost
        FROM cost_logs WHERE healed = 0
    """).fetchone()

    healed = conn.execute("""
        SELECT
            COUNT(*) as count,
            AVG(total_tokens) as avg_tokens,
            AVG(estimated_cost_usd) as avg_cost,
            AVG(latency_ms) as avg_latency,
            SUM(estimated_cost_usd) as total_cost,
            AVG(attempts) as avg_attempts
        FROM cost_logs WHERE healed = 1
    """).fetchone()

    conn.close()

    std_tokens = standard["avg_tokens"] or 0
    heal_tokens = healed["avg_tokens"] or 0
    token_multiplier = round(heal_tokens / max(std_tokens, 1), 2)

    return {
        "standard_queries": {
            "count": standard["count"] or 0,
            "avg_tokens": round(std_tokens),
            "avg_cost_usd": round(standard["avg_cost"] or 0, 6),
            "avg_latency_ms": round(standard["avg_latency"] or 0, 1),
            "total_cost_usd": round(standard["total_cost"] or 0, 4),
        },
        "healed_queries": {
            "count": healed["count"] or 0,
            "avg_tokens": round(heal_tokens),
            "avg_cost_usd": round(healed["avg_cost"] or 0, 6),
            "avg_latency_ms": round(healed["avg_latency"] or 0, 1),
            "total_cost_usd": round(healed["total_cost"] or 0, 4),
            "avg_attempts": round(healed["avg_attempts"] or 0, 1),
        },
        "comparison": {
            "token_multiplier": f"{token_multiplier}×",
            "cost_multiplier": f"{round((healed['avg_cost'] or 0) / max(standard['avg_cost'] or 1, 0.000001), 2)}×",
            "latency_multiplier": f"{round((healed['avg_latency'] or 0) / max(standard['avg_latency'] or 1, 1), 2)}×",
        },
    }


def get_cost_summary() -> dict:
    """Overall cost summary."""
    conn = _get_connection()
    row = conn.execute("""
        SELECT
            COUNT(*) as total_queries,
            SUM(total_tokens) as total_tokens,
            SUM(estimated_cost_usd) as total_cost,
            AVG(total_tokens) as avg_tokens_per_query,
            AVG(estimated_cost_usd) as avg_cost_per_query
        FROM cost_logs
    """).fetchone()
    conn.close()

    return {
        "total_queries": row["total_queries"] or 0,
        "total_tokens_used": row["total_tokens"] or 0,
        "total_cost_usd": round(row["total_cost"] or 0, 4),
        "avg_tokens_per_query": round(row["avg_tokens_per_query"] or 0),
        "avg_cost_per_query_usd": round(row["avg_cost_per_query"] or 0, 6),
    }


# Initialize on import
init_cost_db()
