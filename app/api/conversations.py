"""Conversation session endpoints: list, create, get, delete, export."""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.responses import PlainTextResponse, Response, StreamingResponse

from app.auth import UserPublic, optional_user
from app.dependencies import AppState, get_state
from app.models.conversation import (
    ConversationCreate,
    ConversationSession,
    ExportFormat,
)

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])


def _get_owned(
    session_id: str, state: AppState, user: UserPublic | None
) -> ConversationSession:
    """Fetch a conversation, 404-ing if absent or owned by another user."""
    session = state.conversations.get(session_id)
    if session is None or (user is not None and session.owner_id not in (None, user.id)):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return session


@router.get("", response_model=list[ConversationSession])
def list_conversations(
    state: AppState = Depends(get_state), user: UserPublic | None = Depends(optional_user)
) -> list[ConversationSession]:
    sessions = list(state.conversations.values())
    if user is not None:
        sessions = [s for s in sessions if s.owner_id in (None, user.id)]
    return sessions


@router.post("", response_model=ConversationSession)
def create_conversation(
    body: ConversationCreate,
    state: AppState = Depends(get_state),
    user: UserPublic | None = Depends(optional_user),
) -> ConversationSession:
    session = ConversationSession(
        title=body.title or "New conversation", owner_id=user.id if user else None
    )
    state.conversations[session.id] = session
    return session


@router.get("/{session_id}", response_model=ConversationSession)
def get_conversation(
    session_id: str,
    state: AppState = Depends(get_state),
    user: UserPublic | None = Depends(optional_user),
) -> ConversationSession:
    return _get_owned(session_id, state, user)


@router.delete("/{session_id}")
def delete_conversation(
    session_id: str,
    state: AppState = Depends(get_state),
    user: UserPublic | None = Depends(optional_user),
) -> dict[str, str]:
    _get_owned(session_id, state, user)
    del state.conversations[session_id]
    return {"status": "deleted", "id": session_id}


def _to_markdown(session: ConversationSession) -> str:
    lines = [f"# {session.title}", ""]
    for msg in session.messages:
        lines.append(f"**{msg.role.value.title()}:** {msg.content}")
        for src in msg.sources:
            lines.append(f"> [{src.filename} · p.{src.page_number}] {src.text[:200]}")
        lines.append("")
    return "\n".join(lines)


def _to_pdf(session: ConversationSession) -> bytes:
    """Render a conversation to a real PDF using reportlab's flowables."""
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title=session.title)
    styles = getSampleStyleSheet()
    source_style = ParagraphStyle(
        "Source", parent=styles["Italic"], leftIndent=18, textColor="#666666", fontSize=8
    )

    def _escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    story: list = [Paragraph(_escape(session.title), styles["Title"]), Spacer(1, 12)]
    for msg in session.messages:
        role_style = ParagraphStyle(
            "Role", parent=styles["BodyText"], alignment=TA_LEFT, spaceBefore=8
        )
        story.append(
            Paragraph(f"<b>{msg.role.value.title()}:</b> {_escape(msg.content)}", role_style)
        )
        for src in msg.sources:
            story.append(
                Paragraph(
                    f"[{_escape(src.filename)} · p.{src.page_number}] "
                    f"{_escape(src.text[:200])}",
                    source_style,
                )
            )
    doc.build(story)
    return buf.getvalue()


@router.get("/{session_id}/export")
def export_conversation(
    session_id: str,
    fmt: ExportFormat = Query(default=ExportFormat.MARKDOWN),
    state: AppState = Depends(get_state),
    user: UserPublic | None = Depends(optional_user),
) -> Response:
    """Export a conversation as Markdown or a rendered PDF."""
    session = _get_owned(session_id, state, user)
    if fmt == ExportFormat.PDF:
        try:
            pdf = _to_pdf(session)
        except Exception as exc:  # noqa: BLE001 — reportlab missing → fail clearly
            raise HTTPException(
                status_code=503, detail=f"PDF export unavailable: {exc}"
            ) from exc
        return StreamingResponse(
            iter([pdf]),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={session_id}.pdf"},
        )
    return PlainTextResponse(
        _to_markdown(session),
        headers={"Content-Disposition": f"attachment; filename={session_id}.md"},
    )
