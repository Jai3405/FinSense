"""
Authentication dependencies for FastAPI endpoints.
Integrates with Clerk JWT verification.
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
import httpx

from app.core.config import settings

security = HTTPBearer()


@dataclass
class ClerkUser:
    """Authenticated user from Clerk JWT."""

    id: str
    email: str
    first_name: str | None = None
    last_name: str | None = None
    image_url: str | None = None
    plan: str = "free"
    email_verified: bool = False


async def verify_clerk_token(token: str) -> dict:
    """
    Verify Clerk JWT token.
    In production, this should verify against Clerk's JWKS endpoint.
    """
    try:
        # For development, decode without verification
        # In production, use Clerk's JWKS to verify
        if settings.ENVIRONMENT == "development":
            # Decode without verification for dev
            payload = jwt.decode(
                token,
                settings.API_SECRET_KEY,
                algorithms=["HS256"],
                options={"verify_signature": False},
            )
        else:
            # In production, verify with Clerk's public keys
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.CLERK_JWT_ISSUER}/.well-known/jwks.json"
                )
                jwks = response.json()

            # Verify token with JWKS
            payload = jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=settings.CLERK_PUBLISHABLE_KEY,
                issuer=settings.CLERK_JWT_ISSUER,
            )

        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> ClerkUser:
    """
    Dependency to get current authenticated user from Clerk JWT.
    """
    token = credentials.credentials

    # Verify and decode token
    payload = await verify_clerk_token(token)

    # Extract user info from payload
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID",
        )

    # Build user object
    return ClerkUser(
        id=user_id,
        email=payload.get("email", ""),
        first_name=payload.get("first_name"),
        last_name=payload.get("last_name"),
        image_url=payload.get("image_url"),
        plan=payload.get("public_metadata", {}).get("plan", "free"),
        email_verified=payload.get("email_verified", False),
    )


async def get_optional_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))
    ],
) -> ClerkUser | None:
    """
    Optional authentication - returns None if not authenticated.
    Useful for endpoints that behave differently for logged-in users.
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


def require_plan(required_plans: list[str]):
    """
    Dependency factory to require specific subscription plans.

    Usage:
        @router.get("/pro-feature", dependencies=[Depends(require_plan(["pro", "premium"]))])
    """

    async def check_plan(user: ClerkUser = Depends(get_current_user)) -> ClerkUser:
        if user.plan not in required_plans:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires one of: {', '.join(required_plans)}. "
                f"Your current plan: {user.plan}",
            )
        return user

    return check_plan
