"""
pipeline/decomposer.py — Query Decomposition (Multi-Hop Reasoning)

WHAT: Breaks complex questions into simpler sub-questions.
WHY: "How does boredom connect to creativity, and how is that different from hyperfocus?"
     is actually 3 questions. Searching for all at once gets muddy results.

APPROACH:
1. Classify: is this question simple (single-hop) or complex (multi-hop)?
2. If complex: decompose into 2-4 sub-questions
3. Search each sub-question independently
4. Fuse results using Reciprocal Rank Fusion (RRF)
5. Generate answer from combined context

BASED ON: IRCoT (Trivedi et al, 2023), ReAct (Yao et al, 2022)
"""

from groq import AsyncGroq
from app.config import settings


CLASSIFY_PROMPT = """Classify this question as SIMPLE or COMPLEX.

SIMPLE: can be answered from a single passage (who, what, when, where, define)
COMPLEX: requires combining information from multiple passages, comparing, or multi-step reasoning

Respond with ONLY "SIMPLE" or "COMPLEX"."""


DECOMPOSE_PROMPT = """Break this complex question into 2-4 simpler sub-questions.
Each sub-question should be self-contained and searchable independently.

RULES:
- Each sub-question targets ONE specific piece of information
- Together, the sub-questions cover the full original question
- Keep them short and specific
- Return ONLY the sub-questions, one per line, numbered 1-4

EXAMPLE:
Original: "How does the author's view on boredom compare to their view on distraction, and which is more important for creativity?"
Sub-questions:
1. What does the author say about boredom and its relationship to creativity?
2. What does the author say about distraction?
3. How does the author compare boredom and distraction?
4. Which does the author consider more important for creativity?"""


async def classify_question(question: str) -> str:
    """Classify a question as SIMPLE or COMPLEX."""
    client = AsyncGroq(api_key=settings.groq_api_key)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    result = response.choices[0].message.content.strip().upper()
    return "COMPLEX" if "COMPLEX" in result else "SIMPLE"


async def decompose_question(question: str) -> list[str]:
    """
    Decompose a complex question into sub-questions.

    Returns:
    --------
    list[str] — 2-4 sub-questions. Returns [original] if decomposition fails.
    """
    client = AsyncGroq(api_key=settings.groq_api_key)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": DECOMPOSE_PROMPT},
            {"role": "user", "content": f"Original: {question}\nSub-questions:"},
        ],
        temperature=0.2,
        max_tokens=300,
    )

    text = response.choices[0].message.content.strip()
    lines = text.split("\n")

    sub_questions = []
    for line in lines:
        cleaned = line.strip().lstrip("0123456789.)- ").strip()
        if cleaned and len(cleaned) > 10:  # Skip garbage/short lines
            sub_questions.append(cleaned)

    # Fallback to original if decomposition failed
    if not sub_questions:
        return [question]

    return sub_questions[:4]  # Max 4 sub-questions


async def should_decompose(question: str) -> bool:
    """
    Quick heuristic + LLM check for whether to decompose.
    Heuristic first (free), LLM only if heuristic is uncertain.
    """
    # Heuristic: questions with "and", "compare", "difference", "how...relate"
    # are likely complex
    complex_signals = [
        " and ", " compare", " differ", " versus", " vs ",
        " relate", " connection between", " how does",
        " both ", " contrast",
    ]

    lower_q = question.lower()
    signal_count = sum(1 for s in complex_signals if s in lower_q)

    # Strong signal → classify without LLM
    if signal_count >= 2:
        return True
    if signal_count == 0 and len(question.split()) < 10:
        return False

    # Uncertain → use LLM classifier
    classification = await classify_question(question)
    return classification == "COMPLEX"
