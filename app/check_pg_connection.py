from __future__ import annotations

import json
import os
from urllib.parse import urlparse

import psycopg
from dotenv import load_dotenv


load_dotenv()


def _to_psycopg_dsn(connection: str) -> str:
    # LangChain URLs may use SQLAlchemy-style scheme: postgresql+psycopg2://
    if connection.startswith("postgresql+psycopg2://"):
        return "postgresql://" + connection.split("postgresql+psycopg2://", 1)[1]
    return connection


def _selected_connection() -> tuple[str, str]:
    mode = os.getenv("INGEST_MODE", "raw_sql").strip().lower()
    if mode == "langchain":
        conn = os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
        return mode, _to_psycopg_dsn(conn)
    conn = os.getenv("DATABASE_URL", "").strip()
    return mode, _to_psycopg_dsn(conn)


def check_connection() -> dict:
    mode, connection = _selected_connection()
    if not connection:
        return {
            "status": "error",
            "mode": mode,
            "message": "Missing DB connection string in environment",
            "hint": "Set DATABASE_URL for raw_sql or PGVECTOR_CONNECTION_STRING for langchain",
        }

    parsed = urlparse(connection)
    details = {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/") if parsed.path else "",
        "sslmode_require": "sslmode=require" in connection,
    }

    try:
        with psycopg.connect(connection) as conn:
            with conn.cursor() as cur:
                cur.execute("select current_database(), current_user, version()")
                db_name, db_user, db_version = cur.fetchone()
                cur.execute("select extname from pg_extension where extname = 'vector'")
                vector_enabled = cur.fetchone() is not None
    except Exception as exc:
        return {
            "status": "error",
            "mode": mode,
            "connection": details,
            "message": str(exc),
            "hint": "Check host/port/password/network and run CREATE EXTENSION IF NOT EXISTS vector;",
        }

    return {
        "status": "ok",
        "mode": mode,
        "connection": details,
        "database": {
            "name": db_name,
            "user": db_user,
            "version": db_version,
            "vector_extension_enabled": vector_enabled,
        },
    }


if __name__ == "__main__":
    print(json.dumps(check_connection(), indent=2))
