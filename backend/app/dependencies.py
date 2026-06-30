"""
dependencies.py — Shared FastAPI Dependencies

WHAT ARE DEPENDENCIES:
Reusable functions that FastAPI calls BEFORE your endpoint runs.
If a dependency fails (raises HTTPException), the endpoint never executes.

HOW TO USE:
    from app.dependencies import get_user_id

    @router.post("/upload")
    async def upload(user_id: str = Depends(get_user_id)):
        # user_id is guaranteed to be valid here
        ...

This file provides both strict auth (must be logged in) and
optional auth (works with or without login, for development).
"""

from fastapi import Depends

from app.middleware.auth import get_current_user, get_optional_user


async def get_user_id(user: dict | None = Depends(get_optional_user)) -> str:
    """
    Get the current user's ID.

    - If authenticated: returns real user_id from JWT
    - If not authenticated: returns "default_user" (development mode)

    This allows the app to work both with and without auth configured.
    When you deploy with real NextAuth, all users get proper isolation.
    """
    if user is not None:
        return user["user_id"]
    return "default_user"


async def require_user_id(user: dict = Depends(get_current_user)) -> str:
    """
    Strict version — REQUIRES authentication.
    Returns 401 if not logged in. Use for production endpoints.
    """
    return user["user_id"]
