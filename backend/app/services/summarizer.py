"""
services/summarizer.py — Auto-Summarize Documents on Upload

Generates a one-paragraph TL;DR for each document at ingestion time.
Uses the first few chunks + Groq/Llama 3 to produce a concise summary.
"""

from groq import AsyncGroq
from app.config import settings


SUMMARIZE_PROMPT = """You are a document summarizer. Given the following text excerpts from a document,
write a concise 2-3 sentence summary that captures the main topic and key points.

RULES:
- Be concise (max 3 sentences)
- Focus on WHAT the document is about and its main conclusions/instructions
- Don't start with "This document..." — just state the content directly
- Return ONLY the summary, nothing else"""


async def generate_summary(chunks: list[dict], max_chunks: int = 5) -> str:
    """
    Generate a summary from the first few chunks of a document.

    Parameters:
    -----------
    chunks : list[dict]
        Document chunks (from chunker.py output)
    max_chunks : int
        How many chunks to use for summarization (first N)

    Returns:
    --------
    str — A 2-3 sentence summary of the document
    """
    # Take the first few chunks (beginning of document = most informative)
    sample_chunks = chunks[:max_chunks]
    text = "\n\n".join(c["text"] for c in sample_chunks)

    # Truncate if too long (stay within context limits)
    if len(text) > 4000:
        text = text[:4000] + "..."

    client = AsyncGroq(api_key=settings.groq_api_key)

    try:
        response = await client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SUMMARIZE_PROMPT},
                {"role": "user", "content": f"Document excerpts:\n\n{text}\n\nSummary:"},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Don't let summarization failure break the upload
        return f"Summary unavailable: {str(e)}"
