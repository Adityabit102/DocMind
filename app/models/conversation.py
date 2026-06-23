"""Pydantic schemas for conversation sessions and messages."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.query import SourceChunk


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single turn in a conversation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Role
    content: str
    sources: list[SourceChunk] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationSession(BaseModel):
    """A stored chat session with its full message history."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = "New conversation"
    owner_id: str | None = None
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_cost_usd: float = 0.0
    total_tokens: int = 0


class ConversationCreate(BaseModel):
    title: str | None = None


class ExportFormat(str, Enum):
    PDF = "pdf"
    MARKDOWN = "markdown"
