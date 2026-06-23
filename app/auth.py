"""Lightweight, dependency-free user auth for per-user data isolation.

A JSON-backed user store with PBKDF2-hashed passwords and stateless HMAC-signed
bearer tokens (``<base64 payload>.<hmac sig>``, payload carries the user id and
an expiry). No external auth service or DB — it fits the local-first stack and is
gated by ``ENABLE_AUTH`` so the default single-user experience is unchanged.

Tokens are signed with ``AUTH_SECRET``; set a real secret in production.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from uuid import uuid4

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings, get_settings

_PBKDF2_ROUNDS = 200_000


# ── Models ───────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)


class UserPublic(BaseModel):
    id: str
    username: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


# ── Password hashing ─────────────────────────────────────────────────
def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return base64.b64encode(dk).decode()


def _verify_password(password: str, salt_hex: str, expected: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    return hmac.compare_digest(_hash_password(password, salt), expected)


# ── User store (users.json) ──────────────────────────────────────────
def _load_users() -> dict[str, dict]:
    path = get_settings().users_file
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_users(users: dict[str, dict]) -> None:
    path = get_settings().users_file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(users, fh, indent=2)


def register_user(body: UserCreate) -> UserPublic:
    users = _load_users()
    if any(u["username"] == body.username for u in users.values()):
        raise HTTPException(status_code=409, detail="Username already taken")
    salt = os.urandom(16)
    user_id = str(uuid4())
    users[user_id] = {
        "id": user_id,
        "username": body.username,
        "salt": salt.hex(),
        "password_hash": _hash_password(body.password, salt),
    }
    _save_users(users)
    return UserPublic(id=user_id, username=body.username)


def authenticate(username: str, password: str) -> UserPublic:
    users = _load_users()
    for user in users.values():
        if user["username"] == username and _verify_password(
            password, user["salt"], user["password_hash"]
        ):
            return UserPublic(id=user["id"], username=username)
    raise HTTPException(status_code=401, detail="Invalid username or password")


# ── Stateless tokens ─────────────────────────────────────────────────
def _sign(payload_b64: str, secret: str) -> str:
    sig = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")


def issue_token(user: UserPublic, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    expiry = int(time.time()) + settings.auth_token_ttl_hours * 3600
    payload = json.dumps({"sub": user.id, "name": user.username, "exp": expiry})
    payload_b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"{payload_b64}.{_sign(payload_b64, settings.auth_secret)}"


def verify_token(token: str, settings: Settings | None = None) -> UserPublic | None:
    settings = settings or get_settings()
    try:
        payload_b64, sig = token.split(".", 1)
    except ValueError:
        return None
    if not hmac.compare_digest(sig, _sign(payload_b64, settings.auth_secret)):
        return None
    padded = payload_b64 + "=" * (-len(payload_b64) % 4)
    data = json.loads(base64.urlsafe_b64decode(padded).decode())
    if data.get("exp", 0) < int(time.time()):
        return None
    return UserPublic(id=data["sub"], username=data.get("name", ""))


# ── FastAPI dependencies ─────────────────────────────────────────────
def optional_user(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> UserPublic | None:
    """Resolve the current user from a Bearer token, or None.

    Returns None when auth is disabled (local-first single-user) or no valid
    token is present — callers then operate in unscoped mode.
    """
    if not settings.enable_auth:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return verify_token(authorization.split(" ", 1)[1], settings)


def require_user(
    user: UserPublic | None = Depends(optional_user),
    settings: Settings = Depends(get_settings),
) -> UserPublic | None:
    """Require a valid user when auth is enabled; pass through otherwise."""
    if settings.enable_auth and user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user
