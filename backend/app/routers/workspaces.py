"""
routers/workspaces.py — Workspace Management Endpoints

Provides:
- POST /workspaces/ — Create a workspace
- GET /workspaces/ — List user's workspaces
- POST /workspaces/{id}/members — Add a member
- DELETE /workspaces/{id}/members/{user_id} — Remove a member
- GET /workspaces/{id}/documents — List workspace documents
- DELETE /workspaces/{id} — Delete a workspace
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_user_id
from app.services.workspaces import (
    create_workspace,
    add_member,
    remove_member,
    get_user_workspaces,
    get_workspace_members,
    get_workspace_documents,
    delete_workspace,
    is_member,
)


router = APIRouter(tags=["Workspaces"])


class CreateWorkspaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100, description="Workspace name")
    description: str = Field(default="", max_length=500)


class AddMemberRequest(BaseModel):
    user_id: str = Field(description="User ID to add")
    role: str = Field(default="member", pattern="^(member|viewer)$")


@router.post("/")
async def create_new_workspace(request: CreateWorkspaceRequest, user_id: str = Depends(get_user_id)):
    """Create a new shared workspace. You become the admin."""
    workspace = create_workspace(
        name=request.name,
        description=request.description,
        owner_id=user_id,
    )
    return {"status": "success", "workspace": workspace}


@router.get("/")
async def list_workspaces(user_id: str = Depends(get_user_id)):
    """List all workspaces you belong to."""
    workspaces = get_user_workspaces(user_id)
    return {"workspaces": workspaces}


@router.get("/{workspace_id}")
async def get_workspace(workspace_id: str, user_id: str = Depends(get_user_id)):
    """Get workspace details including members and documents."""
    if not is_member(workspace_id, user_id):
        raise HTTPException(status_code=403, detail="You're not a member of this workspace.")

    members = get_workspace_members(workspace_id)
    documents = get_workspace_documents(workspace_id)

    return {
        "workspace_id": workspace_id,
        "members": members,
        "documents": documents,
        "namespace": f"ws_{workspace_id}",
    }


@router.post("/{workspace_id}/members")
async def add_workspace_member(workspace_id: str, request: AddMemberRequest, user_id: str = Depends(get_user_id)):
    """Add a member to a workspace. Only admins can do this."""
    success = add_member(workspace_id, request.user_id, request.role, requester_id=user_id)
    if not success:
        raise HTTPException(status_code=403, detail="You don't have permission to add members, or user is already a member.")
    return {"status": "success", "message": f"User {request.user_id} added as {request.role}."}


@router.delete("/{workspace_id}/members/{member_id}")
async def remove_workspace_member(workspace_id: str, member_id: str, user_id: str = Depends(get_user_id)):
    """Remove a member from a workspace. Only admins can do this."""
    success = remove_member(workspace_id, member_id, requester_id=user_id)
    if not success:
        raise HTTPException(status_code=403, detail="You don't have permission to remove members.")
    return {"status": "success", "message": f"User {member_id} removed."}


@router.delete("/{workspace_id}")
async def delete_workspace_endpoint(workspace_id: str, user_id: str = Depends(get_user_id)):
    """Delete a workspace. Only the owner can do this."""
    success = delete_workspace(workspace_id, requester_id=user_id)
    if not success:
        raise HTTPException(status_code=403, detail="Only the workspace owner can delete it.")
    return {"status": "success", "message": "Workspace deleted."}
