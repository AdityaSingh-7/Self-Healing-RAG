"""
schemas/documents.py — Request/Response Models for Document Management
"""

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    """Information about an uploaded document."""

    doc_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    pages: int = Field(description="Number of pages parsed")
    chunks: int = Field(description="Number of chunks created")
    ingested_at: str = Field(description="ISO timestamp when document was uploaded")


class IngestResponse(BaseModel):
    """Response after successfully uploading a document."""

    status: str = Field(default="success")
    doc_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    pages_parsed: int = Field(description="Number of pages extracted")
    chunks_created: int = Field(description="Number of chunks generated")
    vectors_stored: int = Field(description="Number of vectors stored in Pinecone")
    message: str = Field(description="Human-readable success message")


class DeleteResponse(BaseModel):
    """Response after deleting a document."""

    status: str = Field(default="success")
    doc_id: str = Field(description="The deleted document's ID")
    message: str = Field(description="Confirmation message")
