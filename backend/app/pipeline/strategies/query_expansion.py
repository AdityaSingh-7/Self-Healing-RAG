"""
strategies/query_expansion.py — Strategy 1: Query Expansion

WHAT: Rewrites the question with more keywords/synonyms.
WHEN: Answer is vague because the original query was too narrow.
COST: 1 extra LLM call (cheap) + 1 re-search

EXAMPLE:
  Original: "Can I carry it over?"
  Expanded: "Can I carry over unused PTO vacation days to next year rollover policy?"
"""

from groq import AsyncGroq
from app.config import settings
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.pipeline.strategies.base import HealingStrategy


EXPANSION_PROMPT = """You are a search query optimizer. Given a question and the issues found with the previous answer,
expand the question by adding relevant keywords, synonyms, and related terms.

RULES:
- Keep the original intent
- Add 3-5 additional relevant terms
- Include both formal and informal variations
- Return ONLY the expanded query, nothing else

EXAMPLE:
Original: "What's the deadline for budget submissions?"
Issues: "Answer discusses fiscal year but not specific deadline"
Expanded: "budget submission deadline due date Q3 Q4 fiscal year cutoff when submit"
"""


class QueryExpansionStrategy(HealingStrategy):
    """Expands the query with synonyms and related terms."""

    @property
    def name(self) -> str:
        return "query_expansion"

    @property
    def description(self) -> str:
        return "Expand query with synonyms and related terms for broader retrieval"

    async def execute(self, question: str, original_results: list[dict], user_id: str, validation_issues: str) -> dict:
        # Step 1: Ask LLM to expand the query
        client = AsyncGroq(api_key=settings.groq_api_key)

        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": EXPANSION_PROMPT},
                {"role": "user", "content": f"Original: {question}\nIssues: {validation_issues}\nExpanded:"},
            ],
            temperature=0.3,
            max_tokens=100,
        )

        expanded_query = response.choices[0].message.content.strip()

        # Step 2: Re-embed and re-search with expanded query
        embedder = EmbeddingService()
        query_embedding = embedder.embed_text(expanded_query)

        store = VectorStoreService()
        results = store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=settings.top_k,
        )

        return {
            "results": results,
            "modified_question": expanded_query,
            "metadata": {
                "original_question": question,
                "expanded_to": expanded_query,
            },
        }
