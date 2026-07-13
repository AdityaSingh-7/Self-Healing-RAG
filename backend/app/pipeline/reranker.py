"""
pipeline/reranker.py — Cross-Encoder Reranking + Late Interaction

TWO-STAGE RETRIEVAL:
Stage 1 (fast): Bi-encoder cosine similarity → top-50 candidates
Stage 2 (precise): Cross-encoder rescores → top-5 final results

WHY TWO STAGES:
- Bi-encoder: O(1) per query (pre-computed embeddings). Fast but imprecise.
- Cross-encoder: O(n) per query (reads query+chunk together). Slow but precise.
- Solution: use bi-encoder to get candidates, cross-encoder to pick the best.

CROSS-ENCODER vs BI-ENCODER:
  Bi-encoder:   embed(query) · embed(chunk) = score    ← independent, can pre-compute
  Cross-encoder: model(query + chunk) = score           ← attends to BOTH, much better

LATE INTERACTION (ColBERT-inspired):
  Instead of one vector per text, get one vector per TOKEN.
  Score = sum of max-similarity between each query token and all chunk tokens.
  Catches fine-grained matches that sentence-level embeddings miss.

REFERENCE: ColBERT (Khattab & Zaharia, 2020), MS MARCO reranking benchmarks
"""

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings


class RerankerService:
    """
    Two-stage reranking: cross-encoder + optional late interaction.

    Stage 1 results (from Pinecone) → cross-encoder rescoring → final top-K.
    """

    _cross_encoder = None
    _token_model = None

    def __init__(self):
        """Load cross-encoder model (lazy, first use only)."""
        if RerankerService._cross_encoder is None:
            # ms-marco-MiniLM is trained specifically for query-document relevance
            # Much better than cosine similarity for ranking
            print("Loading cross-encoder model...")
            RerankerService._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
            )
            print("Cross-encoder loaded!")

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank results using cross-encoder.

        Parameters:
        -----------
        query : str
            The user's question
        results : list[dict]
            Initial retrieval results (from Pinecone)
        top_k : int
            How many to return after reranking

        Returns:
        --------
        list[dict] — reranked results with updated scores
        """
        if not results:
            return results

        # Build query-chunk pairs for cross-encoder
        pairs = [(query, r["text"]) for r in results]

        # Score all pairs (cross-encoder reads both together)
        scores = RerankerService._cross_encoder.predict(pairs)

        # Attach cross-encoder scores to results
        for result, ce_score in zip(results, scores):
            result["cross_encoder_score"] = float(ce_score)
            # Combine: 60% cross-encoder + 40% original (recency-weighted) score
            result["reranked_score"] = (
                0.6 * self._normalize(float(ce_score), scores)
                + 0.4 * result["final_score"]
            )

        # Sort by reranked score
        results.sort(key=lambda x: x["reranked_score"], reverse=True)

        return results[:top_k]

    def late_interaction_score(self, query: str, chunk: str) -> float:
        """
        ColBERT-style late interaction scoring.

        Instead of comparing sentence embeddings, we:
        1. Get per-token embeddings for query
        2. Get per-token embeddings for chunk
        3. For each query token, find max similarity across all chunk tokens
        4. Sum these max similarities

        This catches fine-grained term matches that sentence-level misses.
        """
        if RerankerService._token_model is None:
            # Use the same MiniLM but with output_value="token_embeddings"
            RerankerService._token_model = SentenceTransformer(
                settings.embedding_model
            )

        # Get token-level embeddings
        # Note: sentence-transformers doesn't directly expose token embeddings easily
        # This is a simplified version using word-level chunking
        query_words = query.lower().split()
        chunk_words = chunk.lower().split()[:100]  # Limit chunk tokens

        model = RerankerService._token_model

        # Embed each word (approximation of token-level)
        # In full ColBERT you'd use a special model — this is the concept
        query_vecs = model.encode(query_words)
        chunk_vecs = model.encode(chunk_words) if chunk_words else np.zeros((1, 384))

        # Late interaction: for each query token, find max similarity to any chunk token
        total_score = 0.0
        for q_vec in query_vecs:
            similarities = np.dot(chunk_vecs, q_vec) / (
                np.linalg.norm(chunk_vecs, axis=1) * np.linalg.norm(q_vec) + 1e-8
            )
            total_score += float(np.max(similarities))

        # Normalize by query length
        return total_score / max(len(query_words), 1)

    def _normalize(self, score: float, all_scores) -> float:
        """Normalize a score to [0, 1] range relative to the batch."""
        min_s = float(np.min(all_scores))
        max_s = float(np.max(all_scores))
        if max_s == min_s:
            return 0.5
        return (score - min_s) / (max_s - min_s)
