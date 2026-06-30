"""
parser.py — PDF Text Extraction

WHAT THIS DOES:
Takes a PDF file (as raw bytes) and extracts the text from every page.

HOW IT WORKS:
- PyMuPDF (imported as `fitz`) opens the PDF
- We loop through each page and grab the text
- Returns a list of dicts: [{text: "...", page: 1}, {text: "...", page: 2}, ...]

WHY PYMUPDF:
- Fast (written in C under the hood)
- Handles complex PDFs (columns, tables, headers)
- No external dependencies (like Java for Tika)
- Works with scanned PDFs too (if they have an OCR layer)

WHAT IT CAN'T DO:
- Extract text from images/scans without OCR layer
- Read password-protected PDFs
- Parse complex tables perfectly (no PDF parser can)
"""

import fitz  # PyMuPDF — confusing name, but `fitz` is the import


def parse_pdf(file_bytes: bytes, filename: str) -> list[dict]:
    """
    Extract text from a PDF file.

    Parameters:
    -----------
    file_bytes : bytes
        The raw PDF file contents (from an upload)
    filename : str
        Original filename (stored as metadata for citations later)

    Returns:
    --------
    list[dict]
        Each dict has:
        - "text": the extracted text from that page
        - "page": page number (1-indexed, for human-readable citations)
        - "filename": the source filename

    Example:
    --------
    >>> pages = parse_pdf(pdf_bytes, "handbook.pdf")
    >>> pages[0]
    {"text": "Chapter 1: Introduction...", "page": 1, "filename": "handbook.pdf"}
    """

    # Open the PDF from bytes (not from a file path)
    # fitz.open() can take a file path OR raw bytes with filetype specified
    doc = fitz.open(stream=file_bytes, filetype="pdf")

    pages = []

    # Loop through every page in the document
    for page_num, page in enumerate(doc):
        # page_num starts at 0, but humans count from 1
        # get_text() extracts all text from the page as a single string
        text = page.get_text()

        # Skip empty pages (some PDFs have blank separator pages)
        if text.strip():  # .strip() removes whitespace — empty after strip = skip
            pages.append({
                "text": text.strip(),
                "page": page_num + 1,  # Convert 0-indexed to 1-indexed
                "filename": filename,
            })

    # Always close the document to free memory
    doc.close()

    return pages
