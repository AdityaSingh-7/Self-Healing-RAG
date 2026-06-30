"""
services/llm.py — LLM Service (Groq + Llama 3)

WHAT THIS DOES:
Sends prompts to Groq's Llama 3 API and gets responses.
Supports both regular (wait-for-full-response) and streaming (token-by-token) modes.

GROQ vs OPENAI:
The API is almost identical to OpenAI's — same chat completions format.
The difference is speed (500+ tok/s) and cost ($0.89/M tokens vs $15/M).

STREAMING:
When stream=True, the LLM sends tokens one at a time:
  "Based" → " on" → " the" → " document" → "..."
Instead of waiting 2 seconds for the full response, the user sees
text appear in real-time (~50ms perceived latency).

SYSTEM PROMPT:
We give the LLM specific instructions:
- Answer ONLY from the provided context (no hallucination)
- Cite sources with [filename, page X]
- Say "I don't know" if the context doesn't contain the answer
This makes the LLM a reliable research assistant, not a creative writer.
"""

from collections.abc import AsyncGenerator

from groq import AsyncGroq

from app.config import settings


# System prompt — instructions for how the LLM should behave
SYSTEM_PROMPT = """You are a helpful research assistant. Your job is to answer questions
based ONLY on the provided context (document chunks retrieved from the user's files).

RULES:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. Cite your sources using [filename, page X] format after each claim.
3. If the context doesn't contain enough information, say:
   "I don't have enough information in your documents to answer this fully."
4. Be concise but thorough. Use bullet points for lists.
5. If multiple documents disagree, mention both and note which is more recent.
6. Never make up information. If unsure, say so.

FORMAT:
- Start with a direct answer
- Follow with supporting details from the context
- End with citations
"""


class LLMService:
    """
    Handles all interactions with Groq's Llama 3 API.

    Usage:
        llm = LLMService()

        # Non-streaming (wait for full response)
        answer = await llm.generate(prompt, context_chunks)

        # Streaming (token by token)
        async for token in llm.generate_stream(prompt, context_chunks):
            print(token, end="")
    """

    def __init__(self):
        """
        Initialize the Groq async client.

        WHY ASYNC:
        - Our FastAPI server is async (handles many requests at once)
        - If we used a sync client, ONE slow LLM call blocks ALL other requests
        - Async means: "while waiting for Groq's response, serve other users"
        """
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    def _build_messages(
        self,
        question: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> list[dict]:
        """
        Build the message array for the chat completion API.

        The format is:
        [
            {"role": "system", "content": "You are a helpful..."},      ← Instructions
            {"role": "user", "content": "prev question"},               ← History
            {"role": "assistant", "content": "prev answer"},            ← History
            {"role": "user", "content": "Context:\n...\n\nQuestion:..."} ← Current
        ]

        Parameters:
        -----------
        question : str
            The user's current question
        context_chunks : list[dict]
            Retrieved chunks with text, filename, page
        history : list[dict] | None
            Previous conversation messages (for multi-turn)
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (if any)
        if history:
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Build the context section from retrieved chunks
        context_text = self._format_context(context_chunks)

        # Build the final user message with context + question
        user_message = f"""CONTEXT (retrieved from your documents):
---
{context_text}
---

QUESTION: {question}

Please answer based on the context above. Cite sources with [filename, page X]."""

        messages.append({"role": "user", "content": user_message})

        return messages

    def _format_context(self, chunks: list[dict]) -> str:
        """
        Format retrieved chunks into a readable context string.

        Each chunk is labeled with its source for citation:
        [handbook.pdf, page 3]:
        "The parental leave policy provides 16 weeks..."
        """
        if not chunks:
            return "No relevant documents found."

        formatted = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            page = metadata.get("page", "?")
            text = chunk.get("text", "")

            formatted.append(
                f"[{filename}, page {page}]:\n{text}"
            )

        return "\n\n".join(formatted)

    async def generate(
        self,
        question: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> str:
        """
        Generate a complete answer (non-streaming).

        Use this for:
        - Evaluation (need full text to score)
        - Background processing
        - Testing

        Returns the full answer as a string.
        """
        messages = self._build_messages(question, context_chunks, history)

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,  # Low temp = more factual, less creative
            max_tokens=1024,
            stream=False,
        )

        return response.choices[0].message.content

    async def generate_stream(
        self,
        question: str,
        context_chunks: list[dict],
        history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Generate answer token-by-token (streaming).

        Use this for:
        - Real-time chat UI (tokens appear as they're generated)
        - Better UX (user doesn't stare at loading spinner)

        Yields:
        -------
        str
            One token at a time (word fragments, punctuation, spaces)

        Usage:
            async for token in llm.generate_stream("What is X?", chunks):
                # Send to frontend via SSE
                yield f"data: {token}\n\n"
        """
        messages = self._build_messages(question, context_chunks, history)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            stream=True,  # This makes it return chunks incrementally
        )

        # Iterate over the stream — each chunk has a tiny piece of the response
        async for chunk in stream:
            # Each chunk has choices[0].delta.content (or None if done)
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content
