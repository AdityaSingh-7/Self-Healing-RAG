"""
routers/documents.py — Document Management Endpoints

WHAT THIS DOES:
- GET /documents/ — List all uploaded documents
- DELETE /documents/{doc_id} — Delete a document and all its chunks
"""

from fastapi import APIRouter, Depends, HTTPException

from app.services.vectorstore import VectorStoreService
from app.schemas.documents import DeleteResponse
from app.dependencies import get_user_id


router = APIRouter(tags=["Documents"])


@router.get("/")
async def list_documents(user_id: str = Depends(get_user_id)):
    """
    List all documents uploaded by the current user.
    Returns basic stats about what's stored in Pinecone.
    """
    store = VectorStoreService()
    docs = store.list_documents(user_id)

    return {
        "status": "success",
        "user_id": user_id,
        "documents": docs,
    }


@router.delete("/{doc_id}")
async def delete_document(doc_id: str, user_id: str = Depends(get_user_id)):
    """
    Delete a document and all its chunks from the vector store.
    """
    store = VectorStoreService()

    try:
        store.delete_document(doc_id, user_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}",
        )

    return DeleteResponse(
        doc_id=doc_id,
        message=f"Document '{doc_id}' and all its chunks have been deleted.",
    )
