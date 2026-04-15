from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any

import psycopg

from app.check_pg_connection import _selected_connection
from app.config import settings


def _column_exists(cursor: psycopg.Cursor[Any], table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        select 1
        from information_schema.columns
        where table_schema = current_schema()
          and table_name = %s
          and column_name = %s
        limit 1
        """,
        (table_name, column_name),
    )
    return cursor.fetchone() is not None


def verify_knowledge(table_name: str | None = None) -> dict[str, Any]:
    target_table = (table_name or settings.pgvector_table).strip() or "validex_knowledge"
    mode, connection = _selected_connection()
    if not connection:
        return {
            "status": "error",
            "mode": mode,
            "table": target_table,
            "message": "Missing database connection string",
        }

    with psycopg.connect(connection) as conn:
        with conn.cursor() as cur:
            cur.execute("select to_regclass(%s)", (target_table,))
            if cur.fetchone()[0] is None:
                return {
                    "status": "error",
                    "mode": mode,
                    "table": target_table,
                    "message": f"Table {target_table} not found",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }

            cur.execute(f"select count(*), count(distinct source_url) from {target_table}")
            total_chunks, total_sources = cur.fetchone()

            cur.execute(
                f"select coalesce(topic, 'unknown') as topic, count(*)::int from {target_table} group by topic order by count(*) desc, topic asc"
            )
            topic_rows = cur.fetchall()

            has_provider_column = _column_exists(cur, target_table, "embedding_provider")
            provider_rows: list[tuple[str, int]] = []
            if has_provider_column:
                cur.execute(
                    f"""
                    select coalesce(embedding_provider, 'unknown') as embedding_provider, count(*)::int
                    from {target_table}
                    group by coalesce(embedding_provider, 'unknown')
                    order by count(*) desc, embedding_provider asc
                    """
                )
                provider_rows = [(str(provider or "unknown"), int(count)) for provider, count in cur.fetchall()]

            if has_provider_column:
                cur.execute(f"select count(*) from {target_table} where embedding_provider in ('openai', 'google')")
                genuine_rows = cur.fetchone()[0]
                non_genuine_rows = int(total_chunks) - int(genuine_rows)
                cur.execute(
                    f"select count(*) from {target_table} where coalesce(embedding_provider, 'unknown') = 'fake'"
                )
                fake_rows = cur.fetchone()[0]
            else:
                genuine_rows = 0
                non_genuine_rows = int(total_chunks)
                fake_rows = int(total_chunks)

    total_chunks = int(total_chunks)
    genuine_rows = int(genuine_rows)
    fake_rows = int(fake_rows)
    genuine_pct = round((genuine_rows / total_chunks * 100.0), 2) if total_chunks else 0.0
    non_genuine_pct = round((non_genuine_rows / total_chunks * 100.0), 2) if total_chunks else 0.0

    return {
        "status": "ok",
        "mode": mode,
        "table": target_table,
        "totals": {
            "chunks": total_chunks,
            "sources": int(total_sources),
        },
        "topics": [
            {"topic": str(topic), "chunks": int(chunks)}
            for topic, chunks in topic_rows
        ],
        "embedding_health": {
            "has_provider_column": has_provider_column,
            "genuine_vectors": genuine_rows,
            "non_genuine_vectors": non_genuine_rows,
            "fake_vectors": fake_rows,
            "genuine_percent": genuine_pct,
            "non_genuine_percent": non_genuine_pct,
            "provider_breakdown": [
                {"provider": provider, "chunks": count}
                for provider, count in provider_rows
            ],
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    print(json.dumps(verify_knowledge(), indent=2, ensure_ascii=False))