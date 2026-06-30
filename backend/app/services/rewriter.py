"""
services/rewriter.py — Query Rewriting

WHAT THIS DOES:
Takes a vague follow-up question and rewrites it to be self-contained,
using conversation history for context.

THE PROBLEM:
    History: User asked "What's our PTO policy?" → AI answered.
    Follow-up: "Can I carry it over to next year?"

    If we embed "Can I carry it over to next year?" as-is:
    - "it" doesn't mean anything to the embedding model
    - We'd match chunks about "carrying" things, not PTO

THE SOLUTION:
    Rewrite: "Can I carry over unused PTO days to the next year?"
    Now the embedding model finds PTO-related chunks correctly.

HOW IT WORKS:
    We ask the LLM (Groq/Llama 3) to rewrite the query:
    "Given this conversation history, rewrite the latest question
     to be fully self-contained."

WHEN TO SKIP:
    - First message (no history) → no rewrite needed
    - Question is already self-contained → skip to save latency
"""

from groq import AsyncGroq

from app.config import settings


REWRITE_PROMPT = """You are a query rewriter. Given a conversation history and a follow-up question,
rewrite the follow-up question to be fully self-contained (understandable without the history).

RULES:
- Include all necessary context from the history in the rewritten question
- Keep it as a natural question (not a statement)
- Don't add information that wasn't in the history
- If the question is already self-contained, return it unchanged
- Return ONLY the rewritten question, nothing else (no explanation, no quotes)

EXAMPLES:
History: User: "What's our PTO policy?" Assistant: "20 days per year."
Follow-up: "Can I carry it over?"
Rewritten: "Can I carry over unused PTO days to the next year?"

History: User: "Tell me about the Q4 revenue" Assistant: "Q4 2024 revenue was $4.2B..."
Follow-up: "How does that compare to Q3?"
Rewritten: "How does Q4 2024 revenue of $4.2B compare to Q3 2024 revenue?"
"""


async def rewrite_query(
    question: str,
    history: list[dict],
) -> str:
    """
    Rewrite a follow-up question to be self-contained.

    Parameters:
    -----------
    question : str
        The user's latest question (potentially vague)
    history : list[dict]
        Previous messages [{"role": "user"|"assistant", "content": "..."}]

    Returns:
    --------
    str
        The rewritten question (or original if no rewrite needed)
    """
    # Skip if no history (first message is always self-contained)
    if not history:
        return question

    # Skip if question looks self-contained already (simple heuristic)
    # If it's long enough and doesn't have pronouns like "it", "that", "this"
    vague_indicators = ["it", "that", "this", "they", "them", "those", "there"]
    words = question.lower().split()
    if len(words) > 10 and not any(w in vague_indicators for w in words):
        return question

    # Build the rewrite prompt
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-6:]  # Only use last 3 pairs for rewriting
    )

    messages = [
        {"role": "system", "content": REWRITE_PROMPT},
        {"role": "user", "content": f"History:\n{history_text}\n\nFollow-up: {question}\n\nRewritten:"},
    ]

    # Use Groq to rewrite (fast — ~100ms)
    client = AsyncGroq(api_key=settings.groq_api_key)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0.0,  # Deterministic — we want consistent rewrites
        max_tokens=200,   # Rewrites should be short
    )

    rewritten = response.choices[0].message.content.strip()

    # Safety check: if rewrite is empty or way too different, use original
    if not rewritten or len(rewritten) > len(question) * 5:
        return question

    return rewritten
