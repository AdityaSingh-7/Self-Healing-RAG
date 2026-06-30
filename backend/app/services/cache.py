"""
services/cache.py — Semantic Cache

If a user asks a question that's very similar to a previous question
(cosine similarity > 0.95), return the cached answer instantly.

WHY:
- Same question asked twice = same answer. No need to hit Pinecone + Groq again.
- Saves ~2 seconds of latency and API costs.
- Common in production RAG systems.

HOW:
- Store (question_embedding, answer, sources) in memory
- On new query: compare against all cached embeddings
- If similarity > threshold: return cached answer
- Cache expires after 1 hour (stale answers are worse than slow answers)
"""

import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    """A single cached query + answer."""
    question: str
    embedding: list[float]
    answer: str
    sources: list[dict]
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    """
    In-memory semantic cache for query results.

    Usage:
        cache = SemanticCache()

        # Check cache before running pipeline
        hit = cache.get(query_embedding)
        if hit:
            return hit  # Instant response!

        # After running pipeline, store result
        cache.put(question, query_embedding, answer, sources)
    """

    def __init__(self, threshold: float = 0.95, ttl_seconds: int = 3600, max_entries: int = 1000):
        """
        Parameters:
        -----------
        threshold : float
            Cosine similarity threshold to consider a cache hit (0.95 = very similar)
        ttl_seconds : int
            Time-to-live in seconds (3600 = 1 hour)
        max_entries : int
            Maximum cache entries (LRU eviction when exceeded)
        """
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._cache: list[CacheEntry] = []

    def get(self, query_embedding: list[float]) -> dict | None:
        """
        Check if a similar query exists in cache.

        Returns:
        --------
        dict with {answer, sources, cached_question} if hit, else None
        """
        self._evict_expired()

        if not self._cache:
            return None

        query_vec = np.array(query_embedding)

        for entry in self._cache:
            entry_vec = np.array(entry.embedding)
            # Cosine similarity
            similarity = np.dot(query_vec, entry_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(entry_vec)
            )

            if similarity >= self.threshold:
                return {
                    "answer": entry.answer,
                    "sources": entry.sources,
                    "cached_question": entry.question,
                    "cache_similarity": float(similarity),
                }

        return None

    def put(self, question: str, embedding: list[float], answer: str, sources: list[dict]):
        """Store a query result in cache."""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_entries:
            self._cache.pop(0)

        self._cache.append(CacheEntry(
            question=question,
            embedding=embedding,
            answer=answer,
            sources=sources,
        ))

    def _evict_expired(self):
        """Remove entries older than TTL."""
        now = time.time()
        self._cache = [
            entry for entry in self._cache
            if (now - entry.timestamp) < self.ttl_seconds
        ]

    def clear(self):
        """Clear all cache entries."""
        self._cache = []

    @property
    def size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)


# Global singleton
semantic_cache = SemanticCache()
