from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    GoogleGenerativeAIEmbeddings = None

from app.config import settings


load_dotenv()

_SAFE_TABLE_NAME_RE = re.compile(r'^[a-z][a-z0-9_]{0,62}$')


def _validate_table_name(name: str) -> str:
    """Raise ValueError if name is not a safe PostgreSQL identifier."""
    if not _SAFE_TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table name {name!r}. "
            "Must start with a lowercase letter and contain only [a-z0-9_] (max 63 chars)."
        )
    return name


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


def _build_embedding_client(google_output_dimensionality: int | None = None) -> tuple[Any | None, str]:
    provider = os.getenv("EMBEDDING_PROVIDER", settings.embedding_provider).strip().lower()
    if provider not in {"auto", "openai", "google"}:
        provider = "auto"

    resolved_google_output_dimensionality = google_output_dimensionality
    if resolved_google_output_dimensionality is None:
        env_dimension = os.getenv("GOOGLE_EMBEDDING_OUTPUT_DIMENSION", "").strip()
        if env_dimension:
            resolved_google_output_dimensionality = int(env_dimension)

    if resolved_google_output_dimensionality is not None:
        if resolved_google_output_dimensionality <= 0 or resolved_google_output_dimensionality > 8192:
            raise RuntimeError("GOOGLE_EMBEDDING_OUTPUT_DIMENSION must be in range 1..8192")

    preferred_google_embedding = settings.google_embedding_model.strip()
    if not preferred_google_embedding:
        fallback_google_embedding = settings.embedding_model.strip()
        if fallback_google_embedding.startswith("models/"):
            preferred_google_embedding = fallback_google_embedding
    if not preferred_google_embedding:
        preferred_google_embedding = "models/gemini-embedding-001"
    if not preferred_google_embedding.startswith("models/"):
        preferred_google_embedding = f"models/{preferred_google_embedding}"

    if provider in {"auto", "google"} and settings.google_api_key and GoogleGenerativeAIEmbeddings is not None:
        try:
            google_kwargs: dict[str, Any] = {
                "model": preferred_google_embedding,
                "google_api_key": settings.google_api_key,
            }
            if resolved_google_output_dimensionality is not None:
                google_kwargs["output_dimensionality"] = resolved_google_output_dimensionality

            return (
                GoogleGenerativeAIEmbeddings(**google_kwargs),
                "google",
            )
        except Exception:
            if provider == "google":
                return (None, "")

    if provider in {"auto", "openai"} and settings.openai_api_key and OpenAIEmbeddings is not None:
        try:
            return (
                OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key,
                ),
                "openai",
            )
        except Exception:
            return (None, "")

    return (None, "")


def _embed_records(records: list[dict[str, Any]]) -> tuple[list[list[float]], int, str]:
    embedding_client, _provider = _build_embedding_client()
    if embedding_client is None:
        allow_fake = os.getenv("ALLOW_FAKE_EMBEDDINGS", "0") == "1"
        if not allow_fake:
            raise RuntimeError("No embedding provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
        fake_dim = int(os.getenv("FAKE_EMBEDDING_DIM", "1536"))
        if fake_dim <= 0 or fake_dim > 8192:
            raise RuntimeError("FAKE_EMBEDDING_DIM must be in range 1..8192")
        vectors: list[list[float]] = []
        for item in records:
            text = str(item.get("text", ""))
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed)
            vectors.append([rng.uniform(-1.0, 1.0) for _ in range(fake_dim)])
        return vectors, fake_dim, "fake"

    texts = [str(item.get("text", "")) for item in records]
    vectors = embedding_client.embed_documents(texts)
    if not vectors:
        return [], 0, _provider or "unknown"
    return vectors, len(vectors[0]), _provider or "unknown"


def ingest_jsonl_to_pgvector(
    jsonl_path: str = "data/canonical/au_blog_chunks.jsonl",
    table_name: str = settings.pgvector_table,
    state_path: str = "data/canonical/embedding_state.json",
    incremental: bool = True,
) -> dict[str, Any]:
    table_name = _validate_table_name(table_name)  # Guard against SQL injection
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
            record = dict(record)
            record["chunk_hash"] = signature
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
    embedding_provider = "unknown"
    if changed_records:
        vectors, dimension, embedding_provider = _embed_records(changed_records)
        if not vectors or dimension <= 0:
            return {"status": "error", "message": "embedding failed", "upserted": 0}

    db_url = _database_url()
    safe_dimension = int(dimension) if dimension else 1536
    if safe_dimension <= 0 or safe_dimension > 8192:
        raise RuntimeError("Invalid embedding dimension")

    deleted_count = 0

    # Disable auto-prepared statements for compatibility with transaction poolers.
    with psycopg.connect(db_url, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    chunk_id TEXT PRIMARY KEY,
                    chunk_hash TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL DEFAULT 'unknown',
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
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS chunk_hash TEXT")
            cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS embedding_provider TEXT")

            upsert_sql = f"""
                INSERT INTO {table_name} (
                    chunk_id,
                    chunk_hash,
                    embedding_provider,
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
                    %(chunk_hash)s,
                    %(embedding_provider)s,
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
                    chunk_hash = EXCLUDED.chunk_hash,
                    embedding_provider = EXCLUDED.embedding_provider,
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
                        "chunk_hash": str(item.get("chunk_hash", "")) or _chunk_hash(item),
                        "embedding_provider": str(item.get("embedding_provider", "")) or embedding_provider,
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

    embedding_client, provider = _build_embedding_client()
    if embedding_client is None:
        raise RuntimeError("No embedding provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

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
        vectorstores_module = importlib.import_module("langchain_community.vectorstores")
        PGVector = getattr(vectorstores_module, "PGVector", None)
        if PGVector is None:
            raise RuntimeError("PGVector class was not found in langchain_community.vectorstores")
    except Exception as exc:
        raise RuntimeError(
            "Missing langchain_community PGVector dependency. Install langchain-community and pgvector."
        ) from exc

    PGVector.from_documents(
        embedding=embedding_client,
        documents=docs_to_insert,
        collection_name=collection_name,
        connection_string=conn,
        use_jsonb=True,
    )

    return {
        "status": "ok",
        "collection": collection_name,
        "upserted": len(docs_to_insert),
        "embedding_provider": provider or "unknown",
        "mode": "langchain_pgvector",
    }


if __name__ == "__main__":
    mode = os.getenv("INGEST_MODE", "raw_sql").strip().lower()
    try:
        if mode == "langchain":
            result = ingest_jsonl_to_postgres_langchain()
        else:
            result = ingest_jsonl_to_pgvector()
    except Exception as exc:
        result = {
            "status": "error",
            "mode": mode,
            "message": str(exc),
            "hint": (
                "Set GOOGLE_API_KEY or OPENAI_API_KEY and DATABASE_URL (raw_sql) or PGVECTOR_CONNECTION_STRING (langchain), "
                "then rerun."
            ),
        }
    print(json.dumps(result, indent=2))
