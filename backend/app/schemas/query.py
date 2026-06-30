"""
schemas/query.py — Request/Response Models for the Query Endpoint

WHAT ARE SCHEMAS:
Schemas define the EXACT shape of data going in and out of your API.
- Request schema: "Your POST body MUST have these fields with these types"
- Response schema: "My response WILL have these fields with these types"

WHY THIS MATTERS:
1. VALIDATION: FastAPI rejects bad requests automatically (no manual checking)
2. DOCUMENTATION: Auto-generated docs show exactly what to send/expect
3. TYPE SAFETY: Your IDE gives autocomplete and catches typos
4. CONTRACT: Frontend devs know exactly what to send without asking you

PYDANTIC BaseModel:
- You define fields with types
- Pydantic validates incoming data against those types
- Invalid data → automatic 422 error with helpful message

Example:
    class QueryRequest(BaseModel):
        question: str           # Required string
        top_k: int = 5         # Optional int, defaults to 5

    If someone sends {"question": 123} → 422 error: "question must be a string"
    If someone sends {} → 422 error: "question is required"
    If someone sends {"question": "hello"} → ✅ works, top_k defaults to 5
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """What the user sends when asking a question."""

    question: str = Field(
        ...,  # ... means REQUIRED (no default)
        min_length=1,
        max_length=2000,
        description="The question to ask about your documents",
        examples=["What is our parental leave policy?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,   # ge = greater than or equal to
        le=20,  # le = less than or equal to
        description="Number of relevant chunks to retrieve (1-20)",
    )
    recency_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="How much to weight recency vs similarity (0.0=pure similarity, 1.0=pure recency)",
    )
    stream: bool = Field(
        default=True,
        description="Whether to stream the response via SSE (Server-Sent Events)",
    )


class SourceChunk(BaseModel):
    """A single retrieved source chunk (shown as citation in the UI)."""

    text: str = Field(description="The chunk content")
    filename: str = Field(description="Source document filename")
    page: int = Field(description="Page number in the source document")
    score: float = Field(description="Raw similarity score (0-1)")
    recency_score: float = Field(description="Recency score (0-1)")
    final_score: float = Field(description="Combined ranking score")


class QueryResponse(BaseModel):
    """What we return after answering a question (non-streaming)."""

    answer: str = Field(description="The LLM-generated answer")
    sources: list[SourceChunk] = Field(description="Retrieved chunks used as context")
    query: str = Field(description="The original question (echo back)")
    model: str = Field(description="Which LLM model generated the answer")


class ConversationMessage(BaseModel):
    """A single message in conversation history."""

    role: str = Field(
        ...,
        pattern="^(user|assistant)$",  # Must be exactly "user" or "assistant"
        description="Who sent this message",
    )
    content: str = Field(description="The message text")


class QueryWithHistoryRequest(QueryRequest):
    """Query with conversation history for multi-turn chat."""

    history: list[ConversationMessage] = Field(
        default=[],
        max_length=20,  # Don't send more than 20 messages as history
        description="Previous messages for context (newest last)",
    )
