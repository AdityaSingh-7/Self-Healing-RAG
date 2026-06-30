"""
services/analytics.py — Query Analytics & Observability

Logs every query to SQLite: latency, tokens, scores, user actions.
Provides endpoints to view system performance over time.
"""

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# SQLite database file (created in project root)
DB_PATH = Path(__file__).parent.parent.parent / "analytics.db"


def _get_connection() -> sqlite3.Connection:
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist. Called once at app startup."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL,
            effective_question TEXT,
            answer_preview TEXT,
            top_k INTEGER,
            recency_weight REAL,
            num_results INTEGER,
            avg_similarity REAL,
            avg_recency REAL,
            latency_ms REAL,
            model TEXT,
            tokens_used INTEGER,
            was_rewritten BOOLEAN DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating INTEGER NOT NULL,
            comment TEXT
        );

        CREATE TABLE IF NOT EXISTS document_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            pages INTEGER,
            chunks INTEGER,
            processing_time_ms REAL
        );
    """)
    conn.commit()
    conn.close()


def log_query(
    user_id: str,
    question: str,
    effective_question: str,
    answer_preview: str,
    top_k: int,
    recency_weight: float,
    num_results: int,
    avg_similarity: float,
    avg_recency: float,
    latency_ms: float,
    model: str,
    tokens_used: int = 0,
    was_rewritten: bool = False,
):
    """Log a query and its metrics."""
    conn = _get_connection()
    conn.execute(
        """INSERT INTO query_logs
        (timestamp, user_id, question, effective_question, answer_preview,
         top_k, recency_weight, num_results, avg_similarity, avg_recency,
         latency_ms, model, tokens_used, was_rewritten)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            user_id, question, effective_question, answer_preview[:500],
            top_k, recency_weight, num_results, avg_similarity, avg_recency,
            latency_ms, model, tokens_used, was_rewritten,
        ),
    )
    conn.commit()
    conn.close()


def log_feedback(user_id: str, question: str, answer: str, rating: int, comment: str = ""):
    """Log user feedback (thumbs up/down)."""
    conn = _get_connection()
    conn.execute(
        "INSERT INTO feedback (timestamp, user_id, question, answer, rating, comment) VALUES (?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), user_id, question, answer, rating, comment),
    )
    conn.commit()
    conn.close()


def log_document(user_id: str, doc_id: str, filename: str, pages: int, chunks: int, processing_time_ms: float):
    """Log a document ingestion event."""
    conn = _get_connection()
    conn.execute(
        "INSERT INTO document_logs (timestamp, user_id, doc_id, filename, pages, chunks, processing_time_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), user_id, doc_id, filename, pages, chunks, processing_time_ms),
    )
    conn.commit()
    conn.close()


def get_analytics_summary(days: int = 7) -> dict:
    """Get summary analytics for the last N days."""
    conn = _get_connection()

    # Query stats
    row = conn.execute("""
        SELECT
            COUNT(*) as total_queries,
            AVG(latency_ms) as avg_latency_ms,
            AVG(avg_similarity) as avg_similarity,
            AVG(num_results) as avg_results,
            SUM(tokens_used) as total_tokens
        FROM query_logs
        WHERE timestamp > datetime('now', ?)
    """, (f'-{days} days',)).fetchone()

    # Feedback stats
    fb = conn.execute("""
        SELECT
            COUNT(*) as total_feedback,
            AVG(rating) as avg_rating,
            SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN rating <= 2 THEN 1 ELSE 0 END) as negative
        FROM feedback
        WHERE timestamp > datetime('now', ?)
    """, (f'-{days} days',)).fetchone()

    # Document stats
    doc = conn.execute("""
        SELECT COUNT(*) as total_docs, SUM(chunks) as total_chunks
        FROM document_logs
        WHERE timestamp > datetime('now', ?)
    """, (f'-{days} days',)).fetchone()

    conn.close()

    return {
        "period_days": days,
        "queries": {
            "total": row["total_queries"] or 0,
            "avg_latency_ms": round(row["avg_latency_ms"] or 0, 1),
            "avg_similarity": round(row["avg_similarity"] or 0, 4),
            "total_tokens": row["total_tokens"] or 0,
        },
        "feedback": {
            "total": fb["total_feedback"] or 0,
            "avg_rating": round(fb["avg_rating"] or 0, 2),
            "positive": fb["positive"] or 0,
            "negative": fb["negative"] or 0,
        },
        "documents": {
            "total_ingested": doc["total_docs"] or 0,
            "total_chunks": doc["total_chunks"] or 0,
        },
    }


def get_recent_queries(limit: int = 20) -> list[dict]:
    """Get the most recent queries with their metrics."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM query_logs ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Initialize database on import
init_db()
