from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

import psycopg
from psycopg import sql

from app.config import settings


def _connection_dsn() -> str:
    dsn = os.getenv("DATABASE_URL", "").strip()
    if dsn:
        return dsn

    alt = os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
    if alt.startswith("postgresql+psycopg2://"):
        return "postgresql://" + alt.split("postgresql+psycopg2://", 1)[1]
    return alt


def fetch_source_analytics(table_name: str | None = None) -> dict[str, Any]:
    table = (table_name or settings.pgvector_table).strip()
    if not table:
        raise RuntimeError("PGVECTOR_TABLE is not configured")

    dsn = _connection_dsn()
    if not dsn:
        raise RuntimeError("DATABASE_URL or PGVECTOR_CONNECTION_STRING is required")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select to_regclass(%s)", (table,))
            exists = cur.fetchone()[0]
            if not exists:
                return {
                    "table": table,
                    "total_chunks": 0,
                    "total_sources": 0,
                    "topics": [],
                    "authority_bands": [],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

            table_identifier = sql.Identifier(table)

            cur.execute(
                sql.SQL(
                    """
                    select coalesce(count(*), 0)::int, coalesce(count(distinct source_url), 0)::int
                    from {table}
                    where approved = true
                    """
                ).format(table=table_identifier)
            )
            total_chunks, total_sources = cur.fetchone()

            cur.execute(
                sql.SQL(
                    """
                    select topic, count(*)::int as chunks
                    from {table}
                    where approved = true
                    group by topic
                    order by chunks desc, topic asc
                    """
                ).format(table=table_identifier)
            )
            topics = [
                {"topic": str(topic or "unknown"), "chunks": int(chunks)}
                for topic, chunks in cur.fetchall()
            ]

            cur.execute(
                sql.SQL(
                    """
                    select
                        case
                            when authority_score >= 0.9 then '0.90-1.00'
                            when authority_score >= 0.8 then '0.80-0.89'
                            when authority_score >= 0.7 then '0.70-0.79'
                            when authority_score >= 0.6 then '0.60-0.69'
                            else '<0.60'
                        end as band,
                        count(distinct source_url)::int as sources,
                        count(*)::int as chunks
                    from {table}
                    where approved = true
                    group by band
                    order by band desc
                    """
                ).format(table=table_identifier)
            )
            authority_bands = [
                {"band": str(band), "sources": int(sources), "chunks": int(chunks)}
                for band, sources, chunks in cur.fetchall()
            ]

    return {
        "table": table,
        "total_chunks": int(total_chunks),
        "total_sources": int(total_sources),
        "topics": topics,
        "authority_bands": authority_bands,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_knowledge_health(table_name: str | None = None) -> dict[str, Any]:
    table = (table_name or settings.pgvector_table).strip()
    if not table:
        raise RuntimeError("PGVECTOR_TABLE is not configured")

    dsn = _connection_dsn()
    if not dsn:
        raise RuntimeError("DATABASE_URL or PGVECTOR_CONNECTION_STRING is required")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select to_regclass(%s)", (table,))
            exists = cur.fetchone()[0]
            if not exists:
                return {
                    "table": table,
                    "total_chunks": 0,
                    "has_embedding_provider": False,
                    "genuine_chunks": 0,
                    "fake_chunks": 0,
                    "other_chunks": 0,
                    "genuine_percent": 0.0,
                    "fake_percent": 0.0,
                    "other_percent": 0.0,
                    "provider_breakdown": [],
                    "ready_for_retrieval": False,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

            table_identifier = sql.Identifier(table)
            cur.execute(sql.SQL("select count(*)::int from {table}").format(table=table_identifier))
            total_chunks = int(cur.fetchone()[0])

            cur.execute(
                """
                select exists(
                    select 1
                    from information_schema.columns
                    where table_schema = current_schema()
                      and table_name = %s
                      and column_name = 'embedding_provider'
                )
                """,
                (table,),
            )
            has_embedding_provider = bool(cur.fetchone()[0])

            if has_embedding_provider:
                cur.execute(
                    sql.SQL(
                        """
                        select coalesce(embedding_provider, 'unknown') as provider, count(*)::int as chunks
                        from {table}
                        group by provider
                        order by chunks desc, provider asc
                        """
                    ).format(table=table_identifier)
                )
                provider_rows = [(str(provider), int(chunks)) for provider, chunks in cur.fetchall()]
                provider_map = {provider: chunks for provider, chunks in provider_rows}
                genuine_chunks = int(provider_map.get("google", 0) + provider_map.get("openai", 0))
                fake_chunks = int(provider_map.get("fake", 0))
                other_chunks = max(0, int(total_chunks) - genuine_chunks - fake_chunks)
            else:
                provider_rows = []
                genuine_chunks = 0
                fake_chunks = total_chunks
                other_chunks = 0

    if total_chunks <= 0:
        genuine_percent = 0.0
        fake_percent = 0.0
        other_percent = 0.0
    else:
        genuine_percent = round((genuine_chunks / total_chunks) * 100.0, 2)
        fake_percent = round((fake_chunks / total_chunks) * 100.0, 2)
        other_percent = round((other_chunks / total_chunks) * 100.0, 2)

    return {
        "table": table,
        "total_chunks": int(total_chunks),
        "has_embedding_provider": has_embedding_provider,
        "genuine_chunks": genuine_chunks,
        "fake_chunks": fake_chunks,
        "other_chunks": other_chunks,
        "genuine_percent": genuine_percent,
        "fake_percent": fake_percent,
        "other_percent": other_percent,
        "provider_breakdown": [
            {"provider": provider, "chunks": chunks}
            for provider, chunks in provider_rows
        ],
        "ready_for_retrieval": genuine_chunks > 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
