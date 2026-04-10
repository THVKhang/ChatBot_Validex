from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import settings


load_dotenv()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for pgvector ingestion")
    return database_url


def _load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    states = payload.get("chunk_hashes", {}) if isinstance(payload, dict) else {}
    return states if isinstance(states, dict) else {}


def _chunk_hash(record: dict[str, Any]) -> str:
    payload = {
        "chunk_id": record.get("chunk_id"),
        "doc_id": record.get("doc_id"),
        "topic": record.get("topic"),
        "title": record.get("title"),
        "authority_score": record.get("authority_score"),
        "approved": record.get("approved"),
        "text": record.get("text"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _embed_records(records: list[dict[str, Any]]) -> tuple[list[list[float]], int]:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    embedding_model = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    texts = [str(item.get("text", "")) for item in records]
    vectors = embedding_model.embed_documents(texts)
    if not vectors:
        return [], 0
    return vectors, len(vectors[0])


def ingest_jsonl_to_pgvector(
    jsonl_path: str = "data/canonical/au_blog_chunks.jsonl",
    table_name: str = "rag_blog_chunks",
    state_path: str = "data/canonical/embedding_state.json",
    incremental: bool = True,
) -> dict[str, Any]:
    path = Path(jsonl_path)
    records = _load_jsonl(path)
    if not records:
        return {"status": "error", "message": "no records found", "upserted": 0}

    state_file = Path(state_path)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    previous_state = _load_state(state_file) if incremental else {}

    changed_records: list[dict[str, Any]] = []
    next_state: dict[str, str] = {}
    for record in records:
        chunk_id = str(record.get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        signature = _chunk_hash(record)
        next_state[chunk_id] = signature
        if not incremental or previous_state.get(chunk_id) != signature:
            changed_records.append(record)

    removed_chunk_ids = [chunk_id for chunk_id in previous_state.keys() if chunk_id not in next_state]

    if incremental and not changed_records and not removed_chunk_ids:
        return {
            "status": "ok",
            "table": table_name,
            "dimension": 0,
            "upserted": 0,
            "changed_records": 0,
            "deleted_records": 0,
            "message": "no changed records",
        }

    vectors: list[list[float]] = []
    dimension = 0
    if changed_records:
        vectors, dimension = _embed_records(changed_records)
        if not vectors or dimension <= 0:
            return {"status": "error", "message": "embedding failed", "upserted": 0}

    db_url = _database_url()
    safe_dimension = int(dimension) if dimension else 1536
    if safe_dimension <= 0 or safe_dimension > 8192:
        raise RuntimeError("Invalid embedding dimension")

    deleted_count = 0

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_domain TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    region TEXT NOT NULL,
                    title TEXT NOT NULL,
                    authority_score DOUBLE PRECISION NOT NULL,
                    approved BOOLEAN NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({safe_dimension}) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            upsert_sql = f"""
                INSERT INTO {table_name} (
                    chunk_id,
                    doc_id,
                    source_url,
                    source_domain,
                    source_type,
                    topic,
                    region,
                    title,
                    authority_score,
                    approved,
                    content,
                    embedding
                ) VALUES (
                    %(chunk_id)s,
                    %(doc_id)s,
                    %(source_url)s,
                    %(source_domain)s,
                    %(source_type)s,
                    %(topic)s,
                    %(region)s,
                    %(title)s,
                    %(authority_score)s,
                    %(approved)s,
                    %(content)s,
                    %(embedding)s::vector
                )
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    doc_id = EXCLUDED.doc_id,
                    source_url = EXCLUDED.source_url,
                    source_domain = EXCLUDED.source_domain,
                    source_type = EXCLUDED.source_type,
                    topic = EXCLUDED.topic,
                    region = EXCLUDED.region,
                    title = EXCLUDED.title,
                    authority_score = EXCLUDED.authority_score,
                    approved = EXCLUDED.approved,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
            """

            payloads = []
            for item, vector in zip(changed_records, vectors):
                payloads.append(
                    {
                        "chunk_id": str(item.get("chunk_id", "")),
                        "doc_id": str(item.get("doc_id", "")),
                        "source_url": str(item.get("source_url", "")),
                        "source_domain": str(item.get("source_domain", "")),
                        "source_type": str(item.get("source_type", "webpage")),
                        "topic": str(item.get("topic", "compliance")),
                        "region": str(item.get("region", "AU")),
                        "title": str(item.get("title", "Untitled")),
                        "authority_score": float(item.get("authority_score", 0.5)),
                        "approved": bool(item.get("approved", True)),
                        "content": str(item.get("text", "")),
                        "embedding": json.dumps(vector),
                    }
                )

            if payloads:
                cur.executemany(upsert_sql, payloads)
            if removed_chunk_ids:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE chunk_id = ANY(%s)",
                    (removed_chunk_ids,),
                )
                deleted_count = cur.rowcount or 0
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_topic ON {table_name}(topic)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_domain ON {table_name}(source_domain)"
            )
            conn.commit()

    state_file.write_text(json.dumps({"chunk_hashes": next_state}, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "ok",
        "table": table_name,
        "dimension": safe_dimension,
        "upserted": len(changed_records),
        "changed_records": len(changed_records),
        "deleted_records": deleted_count,
        "total_records": len(records),
        "state_path": str(state_file),
    }


def ingest_jsonl_to_postgres_langchain(
    jsonl_path: str = "data/canonical/au_blog_chunks.jsonl",
    connection_string: str | None = None,
    collection_name: str = "validex_knowledge",
) -> dict[str, Any]:
    """Ingest JSONL into PostgreSQL using LangChain PGVector vectorstore API."""
    path = Path(jsonl_path)
    records = _load_jsonl(path)
    if not records:
        return {"status": "error", "message": f"no records found in {jsonl_path}", "upserted": 0}

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")

    conn = (
        connection_string
        or os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
        or os.getenv("CONNECTION_STRING", "").strip()
    )
    if not conn:
        raise RuntimeError(
            "PGVECTOR_CONNECTION_STRING (or CONNECTION_STRING) is required for LangChain PGVector ingestion"
        )

    docs_to_insert: list[Document] = []
    for data in records:
        text = str(data.get("text", "")).strip()
        if not text:
            continue
        docs_to_insert.append(
            Document(
                page_content=text,
                metadata={
                    "doc_id": str(data.get("doc_id", "")),
                    "chunk_id": str(data.get("chunk_id", "")),
                    "source_url": str(data.get("source_url", "")),
                    "topic": str(data.get("topic", "general")),
                    "region": str(data.get("region", "AU")),
                    "authority_score": float(data.get("authority_score", 0.8)),
                },
            )
        )

    if not docs_to_insert:
        return {"status": "error", "message": "no non-empty text records to ingest", "upserted": 0}

    try:
        from langchain_community.vectorstores import PGVector
    except Exception as exc:
        raise RuntimeError(
            "Missing langchain_community PGVector dependency. Install langchain-community and pgvector."
        ) from exc

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    PGVector.from_documents(
        embedding=embeddings,
        documents=docs_to_insert,
        collection_name=collection_name,
        connection_string=conn,
        use_jsonb=True,
    )

    return {
        "status": "ok",
        "collection": collection_name,
        "upserted": len(docs_to_insert),
        "mode": "langchain_pgvector",
    }


if __name__ == "__main__":
    mode = os.getenv("INGEST_MODE", "raw_sql").strip().lower()
    if mode == "langchain":
        result = ingest_jsonl_to_postgres_langchain()
    else:
        result = ingest_jsonl_to_pgvector()
    print(json.dumps(result, indent=2))
