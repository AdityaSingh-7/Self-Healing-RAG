"""
chunker.py — Text Chunking

WHAT THIS DOES:
Takes the page-level text from parser.py and splits it into smaller chunks
suitable for embedding and retrieval.

WHY WE CHUNK:
1. Embedding models work best on short text (256-512 tokens)
2. Retrieval precision — a 50-page doc as one vector = useless
   A specific paragraph as one vector = precise matching
3. LLM context window — we send top-K chunks to the LLM,
   smaller chunks = more diverse sources fit in the context

THE CHUNKING STRATEGY:
- SentenceSplitter from LlamaIndex
- Splits on sentence boundaries (not mid-word or mid-sentence)
- chunk_size=512 tokens (sweet spot for MiniLM embeddings)
- chunk_overlap=50 tokens (prevents losing context at boundaries)

OVERLAP EXPLAINED:
  Chunk 1: "...employees receive 20 days PTO. This policy was updated in"
  Chunk 2: "This policy was updated in January 2024 to include..."
                                      ↑ OVERLAP — the boundary context is preserved

WITHOUT OVERLAP:
  Chunk 1: "...employees receive 20 days PTO."
  Chunk 2: "This policy was updated in January 2024 to include..."
  ↑ If someone asks "when was PTO updated?" — chunk 2 alone doesn't mention PTO!
"""

import uuid
from datetime import datetime, timezone

from llama_index.core.node_parser import SentenceSplitter

from app.config import settings


def chunk_pages(
    pages: list[dict],
    doc_id: str | None = None,
    user_id: str = "anonymous",
) -> list[dict]:
    """
    Split parsed pages into chunks with metadata.

    Parameters:
    -----------
    pages : list[dict]
        Output from parser.py — [{text, page, filename}, ...]
    doc_id : str | None
        Unique document ID. Auto-generated if not provided.
    user_id : str
        Who uploaded this doc (for multi-tenancy)

    Returns:
    --------
    list[dict]
        Each dict has:
        - "id": unique chunk ID (for Pinecone)
        - "text": the chunk content
        - "metadata": dict with filename, page, doc_id, user_id, ingested_at
    """

    # Generate a unique document ID if not provided
    # uuid4() creates a random unique string like "a3b2c1d4-..."
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    # Create the splitter with our configured settings
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,      # 512 tokens (from .env)
        chunk_overlap=settings.chunk_overlap,  # 50 tokens overlap
    )

    # Timestamp for recency scoring later
    # We use UTC so times are consistent regardless of timezone
    ingested_at = datetime.now(timezone.utc).isoformat()

    chunks = []

    for page_data in pages:
        # Split this page's text into chunks
        # split_text() returns a list of strings
        text_chunks = splitter.split_text(page_data["text"])

        for i, chunk_text in enumerate(text_chunks):
            # Create a unique ID for this chunk
            # Format: doc_abc123_p3_c2 = document abc123, page 3, chunk 2
            chunk_id = f"{doc_id}_p{page_data['page']}_c{i}"

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "text": chunk_text,  # Store text in metadata too (for retrieval display)
                    "filename": page_data["filename"],
                    "page": page_data["page"],
                    "chunk_index": i,
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "ingested_at": ingested_at,
                },
            })

    return chunks
