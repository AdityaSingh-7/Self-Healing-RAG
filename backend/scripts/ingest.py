"""
scripts/ingest.py — Upload documents for benchmarking

Usage:
    python scripts/ingest.py path/to/file1.pdf path/to/file2.pdf

Uploads each file through the full ingestion pipeline:
  Parse → Chunk → Embed → Store in Pinecone
"""

import sys
import time
sys.path.insert(0, ".")

from app.services.file_parser import parse_file
from app.services.chunker import chunk_pages
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
import uuid


def ingest_file(filepath: str, user_id: str = "benchmark_user"):
    """Ingest a single file."""
    print(f"\n📄 Processing: {filepath}")

    # Read file
    with open(filepath, "rb") as f:
        file_bytes = f.read()

    filename = filepath.split("/")[-1]
    start = time.time()

    # Parse
    pages = parse_file(file_bytes, filename)
    print(f"   Parsed: {len(pages)} pages")

    # Chunk
    doc_id = str(uuid.uuid4())
    chunks = chunk_pages(pages, doc_id=doc_id, user_id=user_id)
    print(f"   Chunked: {len(chunks)} chunks")

    # Embed
    embedder = EmbeddingService()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(chunk_texts)
    print(f"   Embedded: {len(embeddings)} vectors")

    # Store
    store = VectorStoreService()
    stored = store.upsert_chunks(chunks, embeddings, user_id=user_id)
    elapsed = time.time() - start
    print(f"   Stored: {stored} vectors in Pinecone")
    print(f"   ⏱️  Time: {elapsed:.1f}s")

    return {"doc_id": doc_id, "filename": filename, "chunks": len(chunks)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <file1> <file2> ...")
        print("Supported: .pdf .txt .md .docx")
        sys.exit(1)

    files = sys.argv[1:]
    print("=" * 60)
    print(f"INGESTING {len(files)} FILE(S)")
    print("=" * 60)

    results = []
    for filepath in files:
        try:
            result = ingest_file(filepath)
            results.append(result)
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\n" + "=" * 60)
    print(f"✅ DONE — {len(results)} files ingested")
    total_chunks = sum(r["chunks"] for r in results)
    print(f"   Total chunks in Pinecone: {total_chunks}")
    print("=" * 60)
    print("\nNext step: python scripts/benchmark.py")


if __name__ == "__main__":
    main()
