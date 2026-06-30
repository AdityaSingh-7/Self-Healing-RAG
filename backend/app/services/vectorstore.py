"""
vectorstore.py — Pinecone Vector Store with Hybrid Search + Recency

WHAT THIS DOES:
1. UPSERT: Store document chunks as vectors in Pinecone
2. SEARCH: Find the most relevant chunks for a query
3. RERANK: Boost recent documents using exponential decay

PINECONE CONCEPTS:
- Index: A collection of vectors (like a database table)
- Namespace: A partition within an index (we use one per user)
- Vector: A list of numbers + metadata
- Upsert: "Insert or update" — if ID exists, overwrite it

HYBRID SEARCH:
- Dense vectors (MiniLM): capture MEANING ("vacation" ≈ "time off")
- Sparse vectors (BM25): capture exact KEYWORDS ("ISO 27001" = "ISO 27001")
- Combined: best of both worlds

RECENCY SCORING:
Documents decay over time using exponential formula:
  recency_score = e^(-0.693 * age_days / half_life)

With half_life=30 days:
  - Today: 1.0 (full score)
  - 30 days ago: 0.5 (half score)
  - 60 days ago: 0.25 (quarter score)
  - 90 days ago: 0.125

Final ranking:
  final_score = (1 - recency_weight) * similarity + recency_weight * recency
  Default: 80% similarity + 20% recency
"""

import math
from datetime import datetime, timezone

from pinecone import Pinecone, ServerlessSpec

from app.config import settings


class VectorStoreService:
    """
    Manages all interactions with Pinecone.

    Usage:
        store = VectorStoreService()
        store.upsert_chunks(chunks, embeddings, user_id="user123")
        results = store.search(query_embedding, user_id="user123", top_k=5)
    """

    def __init__(self):
        """
        Connect to Pinecone and get (or create) our index.
        Lazy initialization — skips connection if API key is a placeholder.
        """
        self.pc = None
        self.index = None

        # Skip initialization if using placeholder key (development mode)
        if settings.pinecone_api_key.startswith("placeholder"):
            return

        # Initialize the Pinecone client with our API key
        self.pc = Pinecone(api_key=settings.pinecone_api_key)

        # Get our index (must already exist in Pinecone dashboard)
        # If it doesn't exist, create it
        self._ensure_index_exists()
        self.index = self.pc.Index(settings.pinecone_index_name)

    def _ensure_index_exists(self):
        """
        Check if our index exists in Pinecone. If not, create it.

        Index settings:
        - dimension=384: must match our embedding model (MiniLM = 384)
        - metric="cosine": how we measure similarity between vectors
        - serverless: pay-per-use (no always-on infrastructure)
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if settings.pinecone_index_name not in existing_indexes:
            print(f"Creating Pinecone index: {settings.pinecone_index_name}")
            self.pc.create_index(
                name=settings.pinecone_index_name,
                dimension=384,  # MiniLM embedding size
                metric="cosine",  # Cosine similarity (0 to 1)
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1",  # Free tier region
                ),
            )
            print("Index created!")

    def _check_connection(self):
        """Raise an error if Pinecone isn't configured."""
        if self.index is None:
            raise RuntimeError(
                "Pinecone is not configured. Please set a valid PINECONE_API_KEY in .env"
            )

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        user_id: str,
    ) -> int:
        """
        Store chunks and their embeddings in Pinecone.

        Parameters:
        -----------
        chunks : list[dict]
            Output from chunker.py — each has "id", "text", "metadata"
        embeddings : list[list[float]]
            Corresponding vectors from embedder.py (same order as chunks)
        user_id : str
            User namespace — keeps each user's documents separate

        Returns:
        --------
        int
            Number of vectors upserted

        HOW NAMESPACES WORK:
        User A's docs go in namespace "user_A"
        User B's docs go in namespace "user_B"
        When User A searches, they ONLY see their own documents.
        Simple multi-tenancy without complex permission logic.
        """
        self._check_connection()

        # Build the vectors in Pinecone's expected format
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append({
                "id": chunk["id"],
                "values": embedding,  # The 384-dim dense vector
                "metadata": chunk["metadata"],  # Text, filename, page, etc.
            })

        # Upsert in batches of 100 (Pinecone's recommended batch size)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(
                vectors=batch,
                namespace=user_id,  # Each user gets their own namespace
            )

        return len(vectors)

    def search(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int | None = None,
        recency_weight: float | None = None,
    ) -> list[dict]:
        """
        Search for the most relevant chunks.

        Parameters:
        -----------
        query_embedding : list[float]
            384-dim vector of the user's question
        user_id : str
            Search only this user's documents
        top_k : int
            How many results to return (default from settings)
        recency_weight : float
            How much to weight recency vs similarity (0.0 to 1.0)
            0.0 = pure similarity, 1.0 = pure recency

        Returns:
        --------
        list[dict]
            Ranked results, each with:
            - "text": the chunk content
            - "score": raw similarity score
            - "recency_score": how recent the document is
            - "final_score": combined score (used for ranking)
            - "metadata": filename, page, ingested_at, etc.
        """
        self._check_connection()

        if top_k is None:
            top_k = settings.top_k
        if recency_weight is None:
            recency_weight = settings.recency_weight

        # Query Pinecone — get more results than needed (we'll rerank)
        # We fetch 2x top_k so recency reranking has room to promote newer docs
        fetch_k = min(top_k * 3, 50)

        results = self.index.query(
            vector=query_embedding,
            top_k=fetch_k,
            include_metadata=True,  # We need metadata for recency + display
            namespace=user_id,
        )

        # Rerank with recency scoring
        ranked_results = self._rerank_with_recency(
            results.matches, recency_weight
        )

        # Return top_k after reranking
        return ranked_results[:top_k]

    def _rerank_with_recency(
        self, matches: list, recency_weight: float
    ) -> list[dict]:
        """
        Apply recency-weighted reranking to search results.

        The formula:
          final_score = (1 - weight) * similarity + weight * recency

        Example with weight=0.2:
          Old doc (90 days): 0.8 * 0.95 + 0.2 * 0.125 = 0.785
          New doc (2 days):  0.8 * 0.88 + 0.2 * 0.97  = 0.898 ← wins!
        """
        ranked = []

        for match in matches:
            # Get the raw similarity score from Pinecone
            similarity = match.score

            # Calculate recency score from the ingested_at timestamp
            ingested_at = match.metadata.get("ingested_at")
            recency = self._calculate_recency(ingested_at)

            # Combine scores
            final_score = (
                (1 - recency_weight) * similarity
                + recency_weight * recency
            )

            ranked.append({
                "text": match.metadata.get("text", ""),
                "score": similarity,
                "recency_score": recency,
                "final_score": final_score,
                "metadata": match.metadata,
            })

        # Sort by final_score (highest first)
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked

    def _calculate_recency(self, ingested_at: str | None) -> float:
        """
        Calculate a recency score using exponential decay.

        Formula: score = e^(-0.693 * age_days / half_life)

        The math:
        - 0.693 = ln(2) — makes the score exactly 0.5 at one half-life
        - half_life = 30 days (configurable in .env)

        Returns:
        --------
        float between 0.0 (ancient) and 1.0 (just uploaded)
        """
        if ingested_at is None:
            return 0.5  # Unknown age — give neutral score

        try:
            # Parse the ISO timestamp string back into a datetime
            ingested_time = datetime.fromisoformat(ingested_at)
            now = datetime.now(timezone.utc)

            # Calculate age in days
            age_days = (now - ingested_time).total_seconds() / 86400  # 86400 sec/day

            # Exponential decay
            # math.exp() = e^(power)
            decay = math.exp(
                -0.693 * age_days / settings.recency_half_life_days
            )

            return decay

        except (ValueError, TypeError):
            # If timestamp is malformed, return neutral score
            return 0.5

    def delete_document(self, doc_id: str, user_id: str) -> None:
        """
        Delete all chunks belonging to a document.

        Uses metadata filtering to find and delete all vectors
        with the matching doc_id.
        """
        # Pinecone doesn't support delete-by-metadata directly in all tiers
        # We'll use the prefix approach since our IDs start with doc_id
        # e.g., "abc123_p1_c0", "abc123_p1_c1", etc.
        self.index.delete(
            filter={"doc_id": {"$eq": doc_id}},
            namespace=user_id,
        )

    def list_documents(self, user_id: str) -> list[dict]:
        """
        List all unique documents for a user.

        Note: Pinecone doesn't have a native "list unique metadata values" operation.
        We'll query with a dummy vector to get stats, or use the list endpoint.
        For now, we'll track documents separately (in a future improvement).
        """
        # This is a simplified version — in production, you'd use a separate
        # metadata store (like a simple SQLite DB) to track documents
        # For now, we query with a zero vector to get some results with metadata
        stats = self.index.describe_index_stats()
        namespace_stats = stats.namespaces.get(user_id, None)

        if namespace_stats is None:
            return []

        return [{"total_vectors": namespace_stats.vector_count}]
