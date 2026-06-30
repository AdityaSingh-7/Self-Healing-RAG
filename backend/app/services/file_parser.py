"""
services/file_parser.py — Multi-Format File Parser

Supports: PDF, TXT, DOCX, Markdown
Extracts text and returns standardized page-level output.
"""

import fitz  # PyMuPDF for PDF
from pathlib import Path

from app.config import settings


def parse_file(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Parse a file based on its extension.

    Supports: .pdf, .txt, .md, .docx

    Parameters:
    -----------
    file_bytes : bytes
        Raw file content
    filename : str
        Original filename (used to determine format)

    Returns:
    --------
    list[dict] — [{text, page, filename}, ...]
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _parse_pdf(file_bytes, filename)
    elif ext in (".txt", ".md", ".markdown"):
        return _parse_text(file_bytes, filename)
    elif ext == ".docx":
        return _parse_docx(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .pdf, .txt, .md, .docx")


def _parse_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """Parse PDF using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text.strip(),
                "page": page_num + 1,
                "filename": filename,
            })

    doc.close()
    return pages


def _parse_text(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Parse plain text / markdown files.
    Split into "pages" by paragraph groups (~2000 chars each).
    """
    text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        return []

    # Split into page-like chunks (~2000 chars each for consistency with PDF pages)
    pages = []
    chunk_size = 2000
    paragraphs = text.split("\n\n")

    current_chunk = ""
    page_num = 1

    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            pages.append({
                "text": current_chunk.strip(),
                "page": page_num,
                "filename": filename,
            })
            page_num += 1
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para

    # Don't forget the last chunk
    if current_chunk.strip():
        pages.append({
            "text": current_chunk.strip(),
            "page": page_num,
            "filename": filename,
        })

    return pages


def _parse_docx(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Parse DOCX files.
    Uses python-docx if available, falls back to basic XML extraction.
    """
    try:
        from docx import Document
        from io import BytesIO

        doc = Document(BytesIO(file_bytes))
        full_text = "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())

        if not full_text.strip():
            return []

        # Split into page-like chunks
        pages = []
        chunk_size = 2000
        page_num = 1

        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            if chunk.strip():
                pages.append({
                    "text": chunk.strip(),
                    "page": page_num,
                    "filename": filename,
                })
                page_num += 1

        return pages

    except ImportError:
        raise ValueError(
            "DOCX support requires python-docx package. Install with: pip install python-docx"
        )


# Supported file extensions
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}

def is_supported(filename: str) -> bool:
    """Check if a file extension is supported."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS
