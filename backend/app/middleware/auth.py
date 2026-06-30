"""
middleware/auth.py — JWT Authentication

WHAT THIS DOES:
Verifies that incoming requests have a valid JWT token from NextAuth.
Extracts the user_id from the token so each user only sees their own docs.

HOW JWT AUTH WORKS IN OUR SYSTEM:
1. User logs in via NextAuth (Google/GitHub OAuth) on the frontend
2. NextAuth creates a signed JWT token containing user info
3. Frontend sends this token in the Authorization header: "Bearer <token>"
4. Backend (this file) verifies the signature and extracts user_id
5. If valid → request proceeds. If invalid → 401 Unauthorized.

WHY JWT (not sessions):
- Stateless: backend doesn't store any session data
- Scalable: works with multiple backend instances
- Simple: just verify the signature, no database lookup needed

THE SHARED SECRET:
Both frontend (NextAuth) and backend use the same NEXTAUTH_SECRET.
NextAuth signs tokens with this secret. We verify with the same secret.
If someone tampers with the token, the signature won't match → rejected.

FASTAPI DEPENDENCY INJECTION:
We define `get_current_user` as a "dependency" — a function that FastAPI
calls automatically before your endpoint runs. If it raises an exception,
the endpoint never executes (the user gets a 401 error).

Usage in endpoints:
    @router.get("/protected")
    async def protected_route(user: dict = Depends(get_current_user)):
        # `user` is guaranteed to be authenticated here
        print(user["user_id"])  # Safe to use
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from jose import JWTError, jwt

from app.config import settings


# HTTPBearer extracts the token from "Authorization: Bearer <token>"
# It handles the parsing — we just get the token string
security = HTTPBearer(
    auto_error=False  # Don't auto-error; we'll handle missing tokens ourselves
)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict:
    """
    Verify JWT token and return the current user.

    This is a FastAPI DEPENDENCY — add it to any endpoint that needs auth:
        @router.get("/protected")
        async def my_endpoint(user: dict = Depends(get_current_user)):
            print(user["user_id"])

    Returns:
    --------
    dict with:
        - user_id: str (unique identifier)
        - email: str (user's email)
        - name: str (display name, if available)

    Raises:
    -------
    HTTPException 401 if token is missing, expired, or invalid
    """
    # Check if credentials were provided
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        # Decode and verify the JWT token
        # algorithms=["HS256"]: NextAuth uses HMAC-SHA256 by default
        payload = jwt.decode(
            token,
            settings.nextauth_secret,
            algorithms=["HS256"],
        )

        # Extract user info from the token payload
        # NextAuth puts these fields in the JWT:
        user_id = payload.get("sub")  # "sub" = subject (standard JWT claim)
        email = payload.get("email")
        name = payload.get("name")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID.",
            )

        return {
            "user_id": user_id,
            "email": email,
            "name": name,
        }

    except JWTError as e:
        # Token is malformed, expired, or signature doesn't match
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict | None:
    """
    Like get_current_user, but returns None instead of 401 if no token.

    Use this for endpoints that work both with and without auth:
    - Authenticated: use user's namespace
    - Unauthenticated: use "default_user" namespace

    Usage:
        @router.get("/flexible")
        async def my_endpoint(user: dict | None = Depends(get_optional_user)):
            user_id = user["user_id"] if user else "default_user"
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None
