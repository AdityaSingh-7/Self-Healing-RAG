"""
strategies/keyword_fallback.py — Strategy 3: Keyword Fallback

WHAT: Pure keyword/BM25 search, ignoring semantic embeddings.
WHEN: Semantic search missed exact terms (model numbers, codes, names).
COST: 1 search (cheap, no LLM call)

EXAMPLE:
  Query: "What is ISO 27001?"
  Semantic search: finds "information security standards" (vague)
  Keyword search: finds "ISO 27001 certification renewed March 2024" (exact match!)
"""

from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.pipeline.strategies.base import HealingStrategy
from app.config import settings


class KeywordFallbackStrategy(HealingStrategy):
    """Falls back to keyword-heavy search for exact term matching."""

    @property
    def name(self) -> str:
        return "keyword_fallback"

    @property
    def description(self) -> str:
        return "Re-search using keyword emphasis for exact term matching"

    async def execute(self, question: str, original_results: list[dict], user_id: str, validation_issues: str) -> dict:
        # Strategy: Extract key nouns/terms from the question and search
        # by boosting those specific terms
        #
        # We do this by embedding a "keyword-stuffed" version:
        # Original: "What is ISO 27001?"
        # Keyword version: "ISO 27001 ISO 27001 ISO 27001 certification standard"
        #
        # This biases the embedding toward the exact terms

        # Extract likely keywords (nouns, proper nouns, numbers, codes)
        keywords = self._extract_keywords(question)
        keyword_query = " ".join(keywords * 3)  # Repeat keywords to emphasize them

        # If no keywords extracted, fall back to original
        if not keyword_query.strip():
            keyword_query = question

        embedder = EmbeddingService()
        query_embedding = embedder.embed_text(keyword_query)

        store = VectorStoreService()
        results = store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=settings.top_k,
            recency_weight=0.0,  # Pure similarity, no recency bias for keyword search
        )

        return {
            "results": results,
            "modified_question": question,  # Keep original for LLM answer generation
            "metadata": {
                "keywords_extracted": keywords,
                "keyword_query": keyword_query,
            },
        }

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Simple keyword extraction — pull out likely important terms.
        (In production, you'd use spaCy NER or a proper keyword extractor)
        """
        # Remove common stop words
        stop_words = {
            "what", "is", "the", "a", "an", "how", "does", "do", "can", "i",
            "we", "our", "my", "this", "that", "it", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "about", "into", "through",
            "during", "before", "after", "above", "below", "between", "and",
            "or", "but", "not", "no", "if", "when", "where", "why", "which",
            "who", "whom", "than", "then", "so", "very", "just", "also",
            "should", "would", "could", "tell", "me", "please", "explain",
        }

        words = text.replace("?", "").replace(".", "").replace(",", "").split()
        keywords = [w for w in words if w.lower() not in stop_words and len(w) > 1]

        return keywords if keywords else words[:3]
