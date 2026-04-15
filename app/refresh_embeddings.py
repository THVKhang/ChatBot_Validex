from __future__ import annotations

from typing import Any
import json
import re

import psycopg

from app.check_pg_connection import _selected_connection
from app.config import settings
from app.ingest_pgvector import _build_embedding_client


_VECTOR_TYPE_DIM_RE = re.compile(r"vector\((\d+)\)", re.IGNORECASE)


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(item):.8f}" for item in values) + "]"


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "resource_exhausted" in message or "quota" in message


def _table_embedding_dimension(cursor: psycopg.Cursor[Any], table_name: str) -> int | None:
    cursor.execute(
        f"""
        select vector_dims(embedding)
        from {table_name}
        where embedding is not null
        limit 1
        """
    )
    row = cursor.fetchone()
    if row and row[0] is not None:
        return int(row[0])

    cursor.execute(
        """
        select pg_catalog.format_type(a.atttypid, a.atttypmod)
        from pg_catalog.pg_attribute a
        where a.attrelid = %s::regclass
          and a.attname = 'embedding'
          and not a.attisdropped
        limit 1
        """,
        (table_name,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    type_name = str(row[0] or "")
    match = _VECTOR_TYPE_DIM_RE.search(type_name)
    if not match:
        return None
    return int(match.group(1))


def _dimension_mismatch_message(
    table_name: str,
    provider: str,
    expected_dim: int,
    actual_dim: int,
) -> str:
    base = (
        "Embedding dimension mismatch: "
        f"table expects {expected_dim} dims but provider '{provider}' returned {actual_dim}. "
        f"Use an embedding model with {expected_dim} dimensions, or recreate {table_name} with vector({actual_dim})."
    )

    if expected_dim == 1536 and bool(settings.openai_api_key.strip()):
        return (
            base
            + " Quick PowerShell fix: $env:EMBEDDING_PROVIDER='openai'; "
            + "$env:EMBEDDING_MODEL='text-embedding-3-small'; python -m app.refresh_embeddings"
        )
    return base


def _get_total_rows(cursor: psycopg.Cursor[Any], table_name: str, refresh_only_fake: bool) -> int:
    cursor.execute(
        f"""
        select count(*)
        from {table_name}
        where {"coalesce(embedding_provider, 'unknown') not in ('openai', 'google')" if refresh_only_fake else 'true'}
        """
    )
    return int(cursor.fetchone()[0])


def refresh_embeddings(
    table_name: str | None = None,
    batch_size: int = 12,
    refresh_only_fake: bool = True,
) -> dict[str, Any]:
    target_table = (table_name or settings.pgvector_table).strip() or "validex_knowledge"
    mode, connection = _selected_connection()
    if not connection:
        return {"status": "error", "table": target_table, "message": "Missing database connection string"}

    provider = "unknown"

    batch_size = max(1, min(50, int(batch_size)))
    processed = 0
    total = 0
    stopped_for_quota = False
    last_chunk_id = None

    # Disable auto-prepared statements for compatibility with transaction poolers.
    with psycopg.connect(connection, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute(f"alter table {target_table} add column if not exists embedding_provider text")
            total = _get_total_rows(cur, target_table, refresh_only_fake)
            expected_dimension = _table_embedding_dimension(cur, target_table)

            embedding_client, provider_hint = _build_embedding_client(
                google_output_dimensionality=expected_dimension,
            )
            if embedding_client is None:
                raise RuntimeError("No live embedding provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
            provider = provider_hint or "unknown"

            while True:
                if refresh_only_fake:
                    where_clause = "coalesce(embedding_provider, 'unknown') not in ('openai', 'google')"
                else:
                    where_clause = "true"

                if last_chunk_id is None:
                    cur.execute(
                        f"""
                        select chunk_id, content
                        from {target_table}
                        where {where_clause}
                        order by chunk_id asc
                        limit %s
                        """,
                        (batch_size,),
                    )
                else:
                    cur.execute(
                        f"""
                        select chunk_id, content
                        from {target_table}
                        where {where_clause} and chunk_id > %s
                        order by chunk_id asc
                        limit %s
                        """,
                        (last_chunk_id, batch_size),
                    )

                rows = cur.fetchall()
                if not rows:
                    break

                texts = [str(content or "") for _, content in rows]
                try:
                    vectors = embedding_client.embed_documents(texts)
                except Exception as exc:
                    if _is_quota_error(exc):
                        stopped_for_quota = True
                        break
                    raise

                if len(vectors) != len(rows):
                    raise RuntimeError(
                        "Embedding provider returned an unexpected number of vectors "
                        f"({len(vectors)} for {len(rows)} rows)."
                    )

                actual_dimension = len(vectors[0]) if vectors else 0
                if expected_dimension is not None and actual_dimension != expected_dimension:
                    conn.rollback()
                    return {
                        "status": "error",
                        "table": target_table,
                        "mode": mode,
                        "provider": provider,
                        "processed_rows": processed,
                        "total_rows": total,
                        "message": _dimension_mismatch_message(
                            target_table,
                            provider,
                            expected_dimension,
                            actual_dimension,
                        ),
                    }

                updates = []
                for (chunk_id, _content), vector in zip(rows, vectors):
                    updates.append(
                        {
                            "chunk_id": chunk_id,
                            "embedding": _vector_literal([float(item) for item in vector]),
                            "embedding_provider": provider,
                        }
                    )

                # Use single-row execute to avoid psycopg pipeline teardown noise on pooled connections.
                for update in updates:
                    cur.execute(
                        f"""
                        update {target_table}
                        set embedding = %(embedding)s::vector,
                            embedding_provider = %(embedding_provider)s
                        where chunk_id = %(chunk_id)s
                        """,
                        update,
                    )
                conn.commit()
                processed += len(rows)
                last_chunk_id = rows[-1][0]

    progress = round((processed / total * 100.0), 2) if total else 100.0
    return {
        "status": "partial" if stopped_for_quota else "ok",
        "table": target_table,
        "mode": mode,
        "provider": provider,
        "refresh_only_fake": refresh_only_fake,
        "batch_size": batch_size,
        "processed_rows": processed,
        "total_rows": total,
        "progress_percent": progress,
        "stopped_for_quota": stopped_for_quota,
        "message": (
            f"Stopped early because the embedding provider returned 429 after processing {processed}/{total} rows."
            if stopped_for_quota
            else f"Refreshed {processed} rows successfully."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(refresh_embeddings(), indent=2, ensure_ascii=False))