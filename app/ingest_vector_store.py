from __future__ import annotations

import importlib
import json
from pathlib import Path

from langchain_core.documents import Document

from app.config import settings

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    GoogleGenerativeAIEmbeddings = None


def _load_metadata(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    metadata_map: dict[str, dict] = {}
    for item in payload if isinstance(payload, list) else []:
        file_stem = str(item.get("file_stem", "")).strip()
        if file_stem:
            metadata_map[file_stem] = item
    return metadata_map


def _build_vector_store():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")
    if not settings.pinecone_index:
        raise RuntimeError("PINECONE_INDEX is required")

    try:
        pinecone_module = importlib.import_module("langchain_pinecone")
        pinecone_vector_store_cls = getattr(pinecone_module, "PineconeVectorStore", None)
    except Exception as exc:
        raise RuntimeError(
            "Missing langchain_pinecone module. Install a compatible langchain-pinecone version."
        ) from exc

    if pinecone_vector_store_cls is None:
        raise RuntimeError("PineconeVectorStore class was not found in langchain_pinecone module")

    embeddings = None
    provider = settings.embedding_provider.strip().lower()
    if provider not in {"auto", "openai", "google"}:
        provider = "auto"

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
        embeddings = GoogleGenerativeAIEmbeddings(
            model=preferred_google_embedding,
            google_api_key=settings.google_api_key,
        )
    elif provider in {"auto", "openai"} and settings.openai_api_key and OpenAIEmbeddings is not None:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    if embeddings is None:
        raise RuntimeError("No embedding provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

    try:
        return pinecone_vector_store_cls(
            index_name=settings.pinecone_index,
            embedding=embeddings,
            namespace=settings.pinecone_namespace or None,
            pinecone_api_key=settings.pinecone_api_key,
            text_key="text",
        )
    except TypeError:
        return pinecone_vector_store_cls.from_existing_index(
            index_name=settings.pinecone_index,
            embedding=embeddings,
            namespace=settings.pinecone_namespace or None,
            text_key="text",
            pinecone_api_key=settings.pinecone_api_key,
        )


def ingest_processed_docs_to_pinecone(
    processed_dir: str | None = None,
    metadata_path: str | None = None,
) -> dict[str, int | str]:
    processed_root = Path(processed_dir or settings.data_processed_dir)
    metadata_file = Path(metadata_path or settings.metadata_path)

    if not processed_root.exists():
        return {"status": "error", "message": "processed directory not found", "upserted": 0}

    metadata_map = _load_metadata(metadata_file)
    vector_store = _build_vector_store()

    documents: list[Document] = []
    ids: list[str] = []

    for txt_file in sorted(processed_root.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        stem = txt_file.stem
        item_meta = metadata_map.get(stem, {})
        doc_id = str(item_meta.get("id") or f"doc_{stem}")

        metadata = {
            "id": doc_id,
            "doc_id": doc_id,
            "file_stem": stem,
            "source": str(txt_file),
            "title": str(item_meta.get("title", stem.replace("_", " ").title())),
            "topic": str(item_meta.get("topic", "")),
            "document_type": str(item_meta.get("document_type", "")),
            "approved": bool(item_meta.get("approved", True)),
        }
        documents.append(Document(page_content=text, metadata=metadata))
        ids.append(doc_id)

    if not documents:
        return {"status": "ok", "message": "no documents to ingest", "upserted": 0}

    vector_store.add_documents(documents=documents, ids=ids)
    return {
        "status": "ok",
        "message": "vector ingestion completed",
        "upserted": len(documents),
    }


if __name__ == "__main__":
    result = ingest_processed_docs_to_pinecone()
    print(json.dumps(result, indent=2))
