"""
learning/failure_logger.py — Healing Event Logger

Logs every healing attempt to SQLite so we can:
1. Track which strategies work
2. See patterns in failures
3. Feed the adaptive strategy selector
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "healing.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_healing_db():
    """Create healing tables."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS healing_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            strategy_name TEXT NOT NULL,
            confidence_before REAL NOT NULL,
            confidence_after REAL NOT NULL,
            improvement REAL NOT NULL,
            success BOOLEAN NOT NULL
        );

        CREATE TABLE IF NOT EXISTS strategy_stats (
            strategy_name TEXT PRIMARY KEY,
            total_attempts INTEGER DEFAULT 0,
            total_successes INTEGER DEFAULT 0,
            avg_improvement REAL DEFAULT 0.0,
            success_rate REAL DEFAULT 0.0,
            last_used TEXT
        );
    """)
    conn.commit()
    conn.close()


def log_healing_event(
    question: str,
    strategy_name: str,
    confidence_before: float,
    confidence_after: float,
    success: bool,
):
    """Log a healing attempt and update strategy stats."""
    conn = _get_connection()
    now = datetime.now(timezone.utc).isoformat()
    improvement = confidence_after - confidence_before

    # Log the event
    conn.execute(
        """INSERT INTO healing_events
        (timestamp, question, strategy_name, confidence_before, confidence_after, improvement, success)
        VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (now, question, strategy_name, confidence_before, confidence_after, improvement, success),
    )

    # Update strategy stats
    conn.execute("""
        INSERT INTO strategy_stats (strategy_name, total_attempts, total_successes, avg_improvement, success_rate, last_used)
        VALUES (?, 1, ?, ?, ?, ?)
        ON CONFLICT(strategy_name) DO UPDATE SET
            total_attempts = total_attempts + 1,
            total_successes = total_successes + ?,
            avg_improvement = (avg_improvement * (total_attempts - 1) + ?) / total_attempts,
            success_rate = CAST(total_successes AS REAL) / total_attempts,
            last_used = ?
    """, (
        strategy_name, int(success), improvement, float(success), now,
        int(success), improvement, now,
    ))

    conn.commit()
    conn.close()


def get_strategy_stats() -> list[dict]:
    """Get performance stats for all strategies."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM strategy_stats ORDER BY success_rate DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_recent_healing_events(limit: int = 20) -> list[dict]:
    """Get recent healing events for debugging."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM healing_events ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_healing_summary() -> dict:
    """Get overall healing stats."""
    conn = _get_connection()

    total = conn.execute("SELECT COUNT(*) as c FROM healing_events").fetchone()["c"]
    successes = conn.execute("SELECT COUNT(*) as c FROM healing_events WHERE success = 1").fetchone()["c"]
    avg_improvement = conn.execute("SELECT AVG(improvement) as a FROM healing_events WHERE success = 1").fetchone()["a"]

    conn.close()

    return {
        "total_healing_attempts": total,
        "successful_healings": successes,
        "healing_success_rate": round(successes / max(total, 1), 4),
        "avg_confidence_improvement": round(avg_improvement or 0, 4),
    }


# Initialize on import
init_healing_db()
