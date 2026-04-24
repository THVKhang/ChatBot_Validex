"""Persistence layer for chat sessions using PostgreSQL.

Provides functions to save and load SessionManager objects to/from
the ``chat_sessions`` table so conversation history survives server
restarts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.session_manager import ChatTurn, SessionManager

logger = logging.getLogger(__name__)


def _connection_dsn() -> str:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if dsn:
        return dsn
    alt = os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
    if alt.startswith("postgresql+psycopg2://"):
        return "postgresql://" + alt.split("postgresql+psycopg2://", 1)[1]
    return alt


def _ensure_table(dsn: str) -> None:
    """Create the chat_sessions and users tables if they don't exist (idempotent)."""
    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        is_admin BOOLEAN NOT NULL DEFAULT FALSE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                try:
                    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN NOT NULL DEFAULT FALSE")
                except Exception:
                    pass
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        turns JSONB NOT NULL DEFAULT '[]'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                try:
                    cur.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE SET NULL")
                except Exception:
                    pass
            conn.commit()
    except Exception as exc:
        logger.debug("session_store._ensure_table skipped: %s", exc)


_table_ensured = False


def _turns_to_json(session: SessionManager) -> str:
    """Serialize session turns to a JSON string."""
    data = []
    for turn in session.turns:
        data.append({
            "user_prompt": turn.user_prompt,
            "assistant_output": turn.assistant_output,
            "parsed_intent": turn.parsed_intent,
            "parsed_topic": turn.parsed_topic,
            "generated_draft": turn.generated_draft[:500] if turn.generated_draft else "",
        })
    return json.dumps(data, ensure_ascii=False)


def _json_to_turns(raw: Any) -> list[ChatTurn]:
    """Deserialize JSON data into a list of ChatTurn objects."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw, list):
        return []

    turns = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        turns.append(ChatTurn(
            user_prompt=str(item.get("user_prompt", "")),
            assistant_output=str(item.get("assistant_output", "")),
            parsed_intent=str(item.get("parsed_intent", "")),
            parsed_topic=str(item.get("parsed_topic", "")),
            generated_draft=str(item.get("generated_draft", "")),
        ))
    return turns


def save_session(session_id: str, session: SessionManager, user_id: int | None = None) -> bool:
    """Persist a session to PostgreSQL. Returns True on success."""
    dsn = _connection_dsn()
    if not dsn:
        return False

    global _table_ensured
    if not _table_ensured:
        _ensure_table(dsn)
        _table_ensured = True

    turns_json = _turns_to_json(session)

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, turns, updated_at)
                    VALUES (%s, %s, %s::jsonb, NOW())
                    ON CONFLICT (session_id)
                    DO UPDATE SET turns = EXCLUDED.turns, user_id = COALESCE(EXCLUDED.user_id, chat_sessions.user_id), updated_at = NOW()
                """, (session_id, user_id, turns_json))
            conn.commit()
        return True
    except Exception as exc:
        logger.warning("session_store.save_session failed: %s", exc)
        return False


def load_session(session_id: str, user_id: int | None = None) -> SessionManager | None:
    """Load a session from PostgreSQL. Returns None if not found or unauthorized."""
    dsn = _connection_dsn()
    if not dsn:
        return None

    global _table_ensured
    if not _table_ensured:
        _ensure_table(dsn)
        _table_ensured = True

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                if user_id is not None:
                    cur.execute(
                        "SELECT turns FROM chat_sessions WHERE session_id = %s AND user_id = %s",
                        (session_id, user_id),
                    )
                else:
                    cur.execute(
                        "SELECT turns FROM chat_sessions WHERE session_id = %s AND user_id IS NULL",
                        (session_id,),
                    )
                row = cur.fetchone()
                if not row:
                    return None

                turns = _json_to_turns(row[0])
                session = SessionManager(turns=turns)
                return session
    except Exception as exc:
        logger.warning("session_store.load_session failed: %s", exc)
        return None


def list_sessions(limit: int = 50, user_id: int | None = None) -> list[dict[str, Any]]:
    """List recent sessions for the sidebar."""
    dsn = _connection_dsn()
    if not dsn:
        return []

    global _table_ensured
    if not _table_ensured:
        _ensure_table(dsn)
        _table_ensured = True

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                if user_id is not None:
                    cur.execute(
                        "SELECT session_id, updated_at, turns FROM chat_sessions WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                        (user_id, limit),
                    )
                else:
                    cur.execute(
                        "SELECT session_id, updated_at, turns FROM chat_sessions WHERE user_id IS NULL ORDER BY updated_at DESC LIMIT %s",
                        (limit,),
                    )
                rows = cur.fetchall()
                sessions = []
                for row in rows:
                    session_id = row[0]
                    updated_at = row[1]
                    turns = _json_to_turns(row[2])
                    
                    # Trích xuất tiêu đề từ tin nhắn đầu tiên
                    title = "New Chat"
                    if turns and turns[0].user_prompt:
                        title = turns[0].user_prompt[:50] + ("..." if len(turns[0].user_prompt) > 50 else "")
                    
                    sessions.append({
                        "session_id": session_id,
                        "updated_at": updated_at.isoformat() if updated_at else None,
                        "title": title,
                        "turn_count": len(turns)
                    })
                return sessions
    except Exception as exc:
        logger.warning("session_store.list_sessions failed: %s", exc)
        return []


def delete_expired_sessions(ttl_seconds: int = 3600) -> int:
    """Delete sessions older than TTL. Returns count of deleted sessions."""
    dsn = _connection_dsn()
    if not dsn:
        return 0

    try:
        import psycopg
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chat_sessions WHERE updated_at < NOW() - INTERVAL '%s seconds'",
                    (ttl_seconds,),
                )
                deleted = cur.rowcount
            conn.commit()
        if deleted:
            logger.info("session_store.cleanup: deleted %d expired sessions", deleted)
        return deleted
    except Exception as exc:
        logger.warning("session_store.delete_expired_sessions failed: %s", exc)
        return 0
