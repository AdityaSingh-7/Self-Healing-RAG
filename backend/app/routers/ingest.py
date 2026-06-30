"""
routers/ingest.py — Document Ingestion Endpoint (Enhanced)

Features:
- Multi-format support (PDF, TXT, DOCX, Markdown)
- Auto-summarization on upload
- Rate limiting (10 uploads/min)
- Analytics logging
- PDF storage for document preview
"""

import uuid
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.services.file_parser import parse_file, is_supported, SUPPORTED_EXTENSIONS
from app.services.chunker import chunk_pages
from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.summarizer import generate_summary
from app.services.rate_limiter import ingest_limiter
from app.services.analytics import log_document
from app.dependencies import get_user_id


router = APIRouter(tags=["Ingestion"])

# Maximum file size: 20MB
MAX_FILE_SIZE = 20 * 1024 * 1024

# Directory to store uploaded files (for preview feature)
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_user_id),
):
    """
    Upload a document for ingestion into the RAG system.

    Supported formats: PDF, TXT, DOCX, Markdown (.md)

    Pipeline:
    1. Validate file (type, size)
    2. Parse text from document
    3. Chunk into retrieval-sized pieces
    4. Generate embeddings
    5. Store in Pinecone
    6. Auto-generate summary
    7. Save original file (for preview)
    """
    start_time = time.time()

    # ==========================================
    # RATE LIMITING
    # ==========================================
    if not ingest_limiter.allow(user_id):
        remaining = ingest_limiter.reset_time(user_id)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max 10 uploads per minute. Try again in {remaining:.0f}s.",
        )

    # ==========================================
    # VALIDATION
    # ==========================================
    if not is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max 20MB, got {len(file_bytes) / 1024 / 1024:.1f}MB.",
        )

    # ==========================================
    # PARSE
    # ==========================================
    try:
        pages = parse_file(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {str(e)}")

    if not pages:
        raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

    # ==========================================
    # CHUNK
    # ==========================================
    doc_id = str(uuid.uuid4())
    chunks = chunk_pages(pages, doc_id=doc_id, user_id=user_id)

    if not chunks:
        raise HTTPException(status_code=422, detail="Failed to create chunks.")

    # ==========================================
    # EMBED
    # ==========================================
    embedder = EmbeddingService()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_batch(chunk_texts)

    # ==========================================
    # STORE IN PINECONE
    # ==========================================
    store = VectorStoreService()
    vectors_stored = store.upsert_chunks(chunks, embeddings, user_id=user_id)

    # ==========================================
    # SAVE ORIGINAL FILE (for preview)
    # ==========================================
    user_upload_dir = UPLOAD_DIR / user_id
    user_upload_dir.mkdir(exist_ok=True)
    file_path = user_upload_dir / f"{doc_id}_{file.filename}"
    file_path.write_bytes(file_bytes)

    # ==========================================
    # AUTO-SUMMARIZE
    # ==========================================
    try:
        summary = await generate_summary(chunks)
    except Exception:
        summary = "Summary generation failed."

    # ==========================================
    # LOG ANALYTICS
    # ==========================================
    processing_time = (time.time() - start_time) * 1000
    try:
        log_document(user_id, doc_id, file.filename, len(pages), len(chunks), processing_time)
    except Exception:
        pass  # Don't let analytics break the upload

    return {
        "status": "success",
        "doc_id": doc_id,
        "filename": file.filename,
        "pages_parsed": len(pages),
        "chunks_created": len(chunks),
        "vectors_stored": vectors_stored,
        "summary": summary,
        "processing_time_ms": round(processing_time, 1),
        "message": f"Successfully ingested '{file.filename}': "
                   f"{len(pages)} pages → {len(chunks)} chunks → {vectors_stored} vectors.",
    }


@router.get("/preview/{doc_id}/{filename}")
async def preview_document(doc_id: str, filename: str, user_id: str = Depends(get_user_id)):
    """
    Get the original uploaded file for preview/viewing.
    Returns the raw file bytes with appropriate content type.
    """
    from fastapi.responses import FileResponse

    file_path = UPLOAD_DIR / user_id / f"{doc_id}_{filename}"

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Document not found.")

    # Determine content type
    ext = Path(filename).suffix.lower()
    content_types = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    return FileResponse(
        path=str(file_path),
        media_type=content_types.get(ext, "application/octet-stream"),
        filename=filename,
    )


@router.get("/formats")
async def supported_formats():
    """List all supported file formats."""
    return {
        "supported": list(SUPPORTED_EXTENSIONS),
        "max_size_mb": MAX_FILE_SIZE / 1024 / 1024,
    }
