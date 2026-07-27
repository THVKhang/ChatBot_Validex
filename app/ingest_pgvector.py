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

import logging
logger = logging.getLogger(__name__)

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
        "parent_id": record.get("parent_id"),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def summarize_table_via_llm(table_md: str) -> str:
    """Generate a summary of the markdown table using Gemini Flash."""
    import time
    from app.langchain_pipeline import pipeline

    # Build fast LLM if not already built
    llm = getattr(pipeline, "_fast_llm", None) or getattr(pipeline, "_llm", None)
    if not llm:
        try:
            from app.llm.provider import build_resilient_fast_llm
            llm = build_resilient_fast_llm(settings)
        except Exception as build_exc:
            print(f"Failed to build fast LLM in table summarizer: {build_exc}")

    prompt = (
        "Generate a concise, 100-word factual text summary of the following markdown table. "
        "Explain its columns, values, and rules clearly:\n\n"
        f"{table_md}"
    )

    max_retries = 4
    backoff = 6.0
    for attempt in range(max_retries):
        try:
            if llm:
                response = llm.invoke(prompt)
                return getattr(response, "content", str(response)).strip()
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "quota" in exc_str.lower() or "limit" in exc_str.lower() or "ResourceExhausted" in exc_str:
                print(f"Rate limit hit in summarize_table_via_llm (attempt {attempt + 1}/{max_retries}). Sleeping {backoff}s...")
                time.sleep(backoff)
                backoff *= 1.5
                continue
            print(f"Failed to summarize table via LLM: {exc}")
            break
    
    # Fallback summary
    lines = [l.strip() for l in table_md.splitlines() if l.strip()]
    header = lines[0] if lines else ""
    values = ", ".join(lines[2:5]) if len(lines) > 2 else ""
    return f"Table with headers: {header}. Contains data: {values}"


def extract_and_summarize_tables(
    doc_id: str,
    text: str,
    doc_meta: dict,
    table_name: str
) -> tuple[str, list[dict], list[dict]]:
    lines = text.splitlines()
    cleaned_lines = []
    table_children = []
    table_parents = []
    
    current_table_lines = []
    in_table = False
    table_idx = 1
    
    for line in lines:
        if "|" in line:
            in_table = True
            current_table_lines.append(line)
        else:
            if in_table and current_table_lines:
                table_md = "\n".join(current_table_lines)
                parent_id = f"parent_{doc_id}_table_{table_idx}"
                
                summary = summarize_table_via_llm(table_md)
                
                child_text = f"[{doc_meta.get('title', 'Document')} > Table Summary]\n{summary}"
                child_id = f"child_{doc_id}_table_{table_idx}"
                table_children.append({
                    **doc_meta,
                    "chunk_id": child_id,
                    "text": child_text,
                    "parent_id": parent_id,
                    "parent_context": f"{doc_meta.get('title', 'Document')} > Table {table_idx}",
                })
                
                table_parents.append({
                    "parent_id": parent_id,
                    "content": table_md,
                    "doc_id": doc_id,
                    "metadata": json.dumps({"type": "table", "title": f"Table {table_idx}", "source_url": doc_meta.get("source_url", "")})
                })
                
                cleaned_lines.append(f"[[TABLE_PLACEHOLDER_{parent_id}]]")
                current_table_lines = []
                in_table = False
                table_idx += 1
            
            cleaned_lines.append(line)
            
    if in_table and current_table_lines:
        table_md = "\n".join(current_table_lines)
        parent_id = f"parent_{doc_id}_table_{table_idx}"
        summary = summarize_table_via_llm(table_md)
        child_text = f"[{doc_meta.get('title', 'Document')} > Table Summary]\n{summary}"
        child_id = f"child_{doc_id}_table_{table_idx}"
        table_children.append({
            **doc_meta,
            "chunk_id": child_id,
            "text": child_text,
            "parent_id": parent_id,
            "parent_context": f"{doc_meta.get('title', 'Document')} > Table {table_idx}",
        })
        table_parents.append({
            "parent_id": parent_id,
            "content": table_md,
            "doc_id": doc_id,
            "metadata": json.dumps({"type": "table", "title": f"Table {table_idx}", "source_url": doc_meta.get("source_url", "")})
        })
        cleaned_lines.append(f"[[TABLE_PLACEHOLDER_{parent_id}]]")
        
    return "\n".join(cleaned_lines), table_children, table_parents


def split_text_into_paragraphs(text: str, max_chars: int = 1500) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = []
    current_len = 0
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if current_len + len(p) + 2 <= max_chars:
            current.append(p)
            current_len += len(p) + 2
        else:
            if current:
                chunks.append("\n\n".join(current))
            current = [p]
            current_len = len(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def split_non_legal_hierarchical(doc: dict, table_name: str) -> tuple[list[dict], list[dict]]:
    full_text = doc["full_text"]
    doc_id = doc["doc_id"]
    doc_title = doc["title"]
    
    full_text, table_children, table_parents = extract_and_summarize_tables(doc_id, full_text, doc, table_name)
    
    child_records = list(table_children)
    parent_records = list(table_parents)
    
    parts = re.split(r'^##\s+', full_text, flags=re.MULTILINE)
    
    intro_text = parts[0].strip()
    if intro_text:
        parent_id = f"parent_{doc_id}_intro"
        p_rec = {
            "parent_id": parent_id,
            "content": intro_text,
            "doc_id": doc_id,
            "metadata": json.dumps({"title": "Introduction", "source_url": doc.get("source_url", "")})
        }
        parent_records.append(p_rec)
        
        sub_chunks = split_text_into_paragraphs(intro_text, max_chars=1500)
        for idx, sub_text in enumerate(sub_chunks):
            child_text = f"[{doc_title} > Introduction]\n{sub_text}"
            child_id = f"child_{doc_id}_intro_{idx}"
            child_records.append({
                **doc,
                "chunk_id": child_id,
                "text": child_text,
                "parent_id": parent_id,
                "parent_context": f"{doc_title} > Introduction",
            })
            
    for sec_idx, part in enumerate(parts[1:], start=1):
        lines = part.splitlines()
        if not lines:
            continue
        h2_title = lines[0].strip()
        sec_text = "\n".join(lines[1:]).strip()
        if not sec_text:
            continue
            
        parent_id = f"parent_{doc_id}_sec_{sec_idx}"
        p_rec = {
            "parent_id": parent_id,
            "content": f"## {h2_title}\n\n{sec_text}",
            "doc_id": doc_id,
            "metadata": json.dumps({"title": h2_title, "source_url": doc.get("source_url", "")})
        }
        parent_records.append(p_rec)
        
        sub_chunks = split_text_into_paragraphs(sec_text, max_chars=1500)
        for idx, sub_text in enumerate(sub_chunks):
            child_text = f"[{doc_title} > {h2_title}]\n{sub_text}"
            child_id = f"child_{doc_id}_sec_{sec_idx}_{idx}"
            child_records.append({
                **doc,
                "chunk_id": child_id,
                "text": child_text,
                "parent_id": parent_id,
                "parent_context": f"{doc_title} > {h2_title}",
            })
            
    return child_records, parent_records


def split_legal_hierarchical(doc: dict, table_name: str) -> tuple[list[dict], list[dict]]:
    full_text = doc["full_text"]
    doc_id = doc["doc_id"]
    doc_title = doc["title"]
    
    full_text, table_children, table_parents = extract_and_summarize_tables(doc_id, full_text, doc, table_name)
    
    child_records = list(table_children)
    parent_records = list(table_parents)
    
    from app.legal_chunker import chunk_legal_text
    legal_chunks = chunk_legal_text(
        text=full_text,
        act_name=doc.get("act_name", doc_title),
        jurisdiction=doc.get("jurisdiction", "Commonwealth"),
        source_url=doc.get("source_url", "")
    )
    
    section_groups = {}
    for lc in legal_chunks:
        section_key = lc.section_ref or lc.breadcrumb or "general"
        if section_key not in section_groups:
            section_groups[section_key] = []
        section_groups[section_key].append(lc)
        
    for sec_idx, (section_key, chunks_in_group) in enumerate(section_groups.items(), start=1):
        parent_id = f"parent_{doc_id}_leg_{sec_idx}"
        parent_content = "\n\n".join(lc.raw_text for lc in chunks_in_group)
        
        p_rec = {
            "parent_id": parent_id,
            "content": f"### {section_key}\n\n{parent_content}",
            "doc_id": doc_id,
            "metadata": json.dumps({"title": section_key, "source_url": doc.get("source_url", "")})
        }
        parent_records.append(p_rec)
        
        for lc in chunks_in_group:
            child_id = f"child_{doc_id}_leg_{sec_idx}_{lc.chunk_index}"
            child_records.append({
                **doc,
                "chunk_id": child_id,
                "text": lc.text,
                "parent_id": parent_id,
                "parent_context": lc.breadcrumb,
            })
            
    return child_records, parent_records


def build_parent_child_chunks(
    records: list[dict[str, Any]],
    table_name: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process raw records, reconstruct documents, and chunk hierarchically."""
    import re
    
    def _detect_legal_structure(text: str) -> bool:
        text_lower = text.lower()
        if "crimes act" in text_lower or "privacy act" in text_lower or "spent convictions" in text_lower:
            return True
        sections = len(re.findall(r'^\s*(?:Section|s\.?|Sec\.?)\s+\d+', text, re.MULTILINE))
        parts = len(re.findall(r'^\s*Part\s+[IVXLC]+', text, re.MULTILINE | re.IGNORECASE))
        return sections >= 3 or parts >= 2

    docs = {}
    for r in records:
        doc_id = r.get("doc_id", "unknown_doc")
        if doc_id not in docs:
            docs[doc_id] = {
                "doc_id": doc_id,
                "title": r.get("title", "Untitled"),
                "source_url": r.get("source_url", ""),
                "source_domain": r.get("source_domain", ""),
                "source_type": r.get("source_type", "webpage"),
                "topic": r.get("topic", "compliance"),
                "region": r.get("region", "AU"),
                "authority_score": r.get("authority_score", 0.5),
                "approved": r.get("approved", True),
                "jurisdiction": r.get("jurisdiction", "Commonwealth"),
                "act_name": r.get("act_name", ""),
                "section_ref": r.get("section_ref", ""),
                "parent_context": r.get("parent_context", ""),
                "chunks": []
            }
        docs[doc_id]["chunks"].append(r)
        
    all_children = []
    all_parents = []
    
    for doc_id, doc in docs.items():
        def get_chunk_idx(c):
            cid = c.get("chunk_id", "")
            match = re.search(r'_(\d+)$', cid)
            return int(match.group(1)) if match else 0
            
        doc["chunks"].sort(key=get_chunk_idx)
        full_text = "\n\n".join(c.get("text", "") for c in doc["chunks"])
        doc["full_text"] = full_text
        
        doc_meta = {k: v for k, v in doc.items() if k != "chunks"}
        
        try:
            if _detect_legal_structure(full_text):
                children, parents = split_legal_hierarchical(doc_meta, table_name)
            else:
                children, parents = split_non_legal_hierarchical(doc_meta, table_name)
        except Exception as exc:
            print(f"Hierarchical splitting failed for doc {doc_id}: {exc}. Using fallback chunking.")
            parent_id = f"parent_{doc_id}_fallback"
            parents = [{
                "parent_id": parent_id,
                "content": full_text,
                "doc_id": doc_id,
                "metadata": json.dumps({"title": doc_meta.get("title", ""), "source_url": doc_meta.get("source_url", "")})
            }]
            children = []
            for c in doc["chunks"]:
                children.append({
                    **doc_meta,
                    "chunk_id": c.get("chunk_id"),
                    "text": c.get("text"),
                    "parent_id": parent_id,
                    "parent_context": doc_meta.get("title", ""),
                })
        
        all_children.extend(children)
        all_parents.extend(parents)
        
    return all_children, all_parents


def _build_embedding_client(google_output_dimensionality: int | None = None) -> tuple[Any | None, str]:
    provider = os.getenv("EMBEDDING_PROVIDER", settings.embedding_provider).strip().lower()
    
    if provider == "local":
        try:
            from sentence_transformers import SentenceTransformer
            # Prefer fine-tuned Validex domain model if available
            finetuned_path = os.path.join("data", "models", "bge-base-finetuned-validex")
            if os.path.isdir(finetuned_path) and os.path.isfile(os.path.join(finetuned_path, "config.json")):
                model_name = finetuned_path
                print(f"Using FINE-TUNED embedding model for ingestion: {finetuned_path}")
            else:
                model_name = 'BAAI/bge-base-en-v1.5'
                print(f"Using pre-trained embedding model for ingestion: {model_name}")

            class LocalEmbeddings:
                def __init__(self, model_path):
                    self.model = SentenceTransformer(model_path)
                def embed_documents(self, texts):
                    return self.model.encode(texts).tolist()
                def embed_query(self, text):
                    return self.model.encode(text).tolist()
            return (LocalEmbeddings(model_name), "local")
        except ImportError:
            print("Please install sentence-transformers: pip install sentence-transformers")
            return (None, "local")

    if provider not in {"auto", "openai", "google", "fake", "local"}:
        provider = "auto"

    if provider == "fake":
        return (None, "fake")

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

    # Run Hierarchical Chunking (Parent-Child) & Table Summarization
    child_records, parent_records = build_parent_child_chunks(records, table_name)

    changed_records: list[dict[str, Any]] = []
    next_state: dict[str, str] = {}
    for record in child_records:
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

    from app.vector_repository import PGVectorRepository
    repo = PGVectorRepository()
    repo.initialize_schema(table_name, safe_dimension)

    # Upsert parent chunks associated with changed child chunks
    if changed_records:
        changed_parent_ids = {r.get("parent_id") for r in changed_records if r.get("parent_id")}
        parents_to_upsert = [p for p in parent_records if p["parent_id"] in changed_parent_ids]
        if parents_to_upsert:
            repo.upsert_parents(table_name, parents_to_upsert)

    # Enrich records with legal metadata before inserting
    from app.metadata_enricher import enrich_batch
    use_llm_enrichment = os.getenv("ENRICH_WITH_LLM", "0") == "1"
    enrich_batch(changed_records, use_llm=use_llm_enrichment)

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
                "parent_id": item.get("parent_id"),
                # Legal metadata (Trụ Cột 3)
                "status": str(item.get("status", "in_force")),
                "jurisdiction": str(item.get("jurisdiction", "Commonwealth")),
                "document_type": str(item.get("document_type", "webpage")),
                "act_name": str(item.get("act_name", "")),
                "section_ref": str(item.get("section_ref", "")),
                "effective_date": str(item.get("effective_date", "")),
                "parent_context": str(item.get("parent_context", "")),
            }
        )

    if payloads:
        repo.upsert_records(table_name, payloads)
    if removed_chunk_ids:
        deleted_count = repo.delete_records(table_name, removed_chunk_ids)

    state_file.write_text(json.dumps({"chunk_hashes": next_state}, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "status": "ok",
        "table": table_name,
        "dimension": safe_dimension,
        "upserted": len(changed_records),
        "changed_records": len(changed_records),
        "deleted_records": deleted_count,
        "total_records": len(child_records),
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
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Ingest documents into PGVector")
    parser.add_argument(
        "--collection",
        type=str,
        default=settings.pgvector_table,
        help="Target collection/table name (e.g. validex_docs_v2)"
    )
    parser.add_argument(
        "--crawl-golden",
        action="store_true",
        help="Crawl golden sources (.gov.au) before ingesting"
    )
    args = parser.parse_args()
    
    mode = os.getenv("INGEST_MODE", "raw_sql").strip().lower()
    
    jsonl_path = "data/canonical/au_blog_chunks.jsonl"
    if args.crawl_golden:
        logger.info("Crawling golden sources before ingestion...")
        from app.gov_crawler import crawl_golden_sources, CRAWLED_OUTPUT_PATH
        crawl_result = crawl_golden_sources()
        if crawl_result.get("success", 0) == 0:
            logger.error("Crawling failed, aborting ingestion.")
            print(json.dumps({"status": "error", "message": "crawling failed"}, indent=2))
            sys.exit(1)
        jsonl_path = CRAWLED_OUTPUT_PATH

    logger.info("Starting ingestion: mode=%s, collection=%s, file=%s", mode, args.collection, jsonl_path)
    
    try:
        if mode == "langchain":
            result = ingest_jsonl_to_postgres_langchain(jsonl_path=jsonl_path, collection_name=args.collection)
        else:
            result = ingest_jsonl_to_pgvector(jsonl_path=jsonl_path, table_name=args.collection)
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
