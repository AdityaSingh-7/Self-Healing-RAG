"""
pipeline/validator.py — Answer Validation (LLM-as-Judge)

THE CORE OF SELF-HEALING:
After generating an answer, we ask a SECOND LLM call:
"Does this answer actually address the question using evidence from the context?"

Returns a confidence score (0.0 to 1.0) and a reason.

WHY THIS WORKS:
LLMs are better at JUDGING answers than generating them.
It's easier to check "is this answer about PTO policy?" than to
produce the perfect PTO answer from scratch.

BASED ON:
- RAGAS Faithfulness metric
- Microsoft's CRAG paper (2024)
- OpenAI's self-consistency technique
"""

from groq import AsyncGroq
from app.config import settings


VALIDATION_PROMPT = """You are an answer quality judge. Given a question, retrieved context, and a generated answer,
evaluate whether the answer is CORRECT, COMPLETE, and GROUNDED in the context.

Score from 0.0 to 1.0:
- 1.0: Answer directly addresses the question with specific evidence from context
- 0.8: Answer addresses the question but could be more specific
- 0.6: Answer is partially relevant but misses key information
- 0.4: Answer is vague or only tangentially related to the question
- 0.2: Answer doesn't address the question OR contradicts the context
- 0.0: Answer is completely wrong or hallucinated

RESPOND WITH EXACTLY THIS FORMAT (nothing else):
CONFIDENCE: <score>
REASON: <one sentence explaining why>
ISSUES: <what's missing or wrong, or "none">"""


async def validate_answer(
    question: str,
    answer: str,
    context_chunks: list[dict],
) -> dict:
    """
    Validate an answer using LLM-as-judge.

    Parameters:
    -----------
    question : str
        The original user question
    answer : str
        The generated answer to validate
    context_chunks : list[dict]
        The retrieved chunks used as context

    Returns:
    --------
    dict with:
        - confidence: float (0.0 to 1.0)
        - reason: str (why this score)
        - issues: str (what's wrong, or "none")
    """
    context_text = "\n\n".join(
        f"[{c.get('metadata', {}).get('filename', '?')}, page {c.get('metadata', {}).get('page', '?')}]: {c.get('text', '')[:500]}"
        for c in context_chunks[:5]
    )

    prompt = f"""QUESTION: {question}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER: {answer}

Evaluate this answer. Is it correct, complete, and grounded in the context?"""

    client = AsyncGroq(api_key=settings.groq_api_key)

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": VALIDATION_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=150,
        )

        result_text = response.choices[0].message.content.strip()
        return _parse_validation(result_text)

    except Exception as e:
        return {
            "confidence": 0.7,
            "reason": f"Validation error: {str(e)}",
            "issues": "Could not validate",
        }


def _parse_validation(text: str) -> dict:
    """Parse the structured validation response."""
    confidence = 0.7
    reason = "Could not parse validation"
    issues = "unknown"

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = float(line.split(":", 1)[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except ValueError:
                pass
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
        elif line.startswith("ISSUES:"):
            issues = line.split(":", 1)[1].strip()

    return {"confidence": confidence, "reason": reason, "issues": issues}
