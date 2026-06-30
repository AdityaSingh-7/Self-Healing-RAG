"""
routers/export.py — Export Chat & Documents

Provides:
- POST /export/chat — Export conversation history as markdown
- GET /export/document-summary — Export all document summaries
"""

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.dependencies import get_user_id
from app.services.memory import conversation_memory


router = APIRouter(tags=["Export"])


class ExportChatRequest(BaseModel):
    """Request to export chat history."""
    format: str = Field(default="markdown", pattern="^(markdown|json)$")


@router.post("/chat")
async def export_chat(request: ExportChatRequest, user_id: str = Depends(get_user_id)):
    """
    Export the current conversation as markdown or JSON.
    Users can download this for their records.
    """
    history = conversation_memory.get_history(user_id)

    if not history:
        return {"status": "empty", "message": "No conversation history to export."}

    if request.format == "markdown":
        md = "# RAG System — Chat Export\n\n"
        md += "---\n\n"

        for msg in history:
            if msg["role"] == "user":
                md += f"## 🧑 You\n\n{msg['content']}\n\n"
            else:
                md += f"## 🤖 Assistant\n\n{msg['content']}\n\n---\n\n"

        return PlainTextResponse(
            content=md,
            media_type="text/markdown",
            headers={"Content-Disposition": "attachment; filename=chat-export.md"},
        )
    else:
        return {
            "format": "json",
            "messages": history,
            "message_count": len(history),
        }
