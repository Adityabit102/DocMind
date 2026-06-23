"""Authentication endpoints: register, login, current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.auth import (
    TokenResponse,
    UserCreate,
    UserPublic,
    authenticate,
    issue_token,
    register_user,
    require_user,
)
from app.config import Settings, get_settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _require_enabled(settings: Settings) -> None:
    if not settings.enable_auth:
        raise HTTPException(status_code=404, detail="Authentication is disabled")


@router.post("/register", response_model=TokenResponse)
def register(body: UserCreate, settings: Settings = Depends(get_settings)) -> TokenResponse:
    _require_enabled(settings)
    user = register_user(body)
    return TokenResponse(access_token=issue_token(user, settings), user=user)


@router.post("/login", response_model=TokenResponse)
def login(body: UserCreate, settings: Settings = Depends(get_settings)) -> TokenResponse:
    _require_enabled(settings)
    user = authenticate(body.username, body.password)
    return TokenResponse(access_token=issue_token(user, settings), user=user)


@router.get("/me", response_model=UserPublic)
def me(user: UserPublic | None = Depends(require_user)) -> UserPublic:
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user
