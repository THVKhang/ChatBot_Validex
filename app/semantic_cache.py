"""Semantic caching layer using PostgreSQL pgvector."""

import json
import logging
import threading
from typing import Any

from app.config import settings
from app.db_pool import _get_dsn, get_connection

logger = logging.getLogger(__name__)

CACHE_TABLE = "validex_semantic_cache"


class PgSemanticCache:
    """Semantic cache using PostgreSQL pgvector for fast similarity search."""

    def __init__(self, threshold: float | None = None):
        self.threshold = (
            threshold if threshold is not None else float(settings.semantic_cache_threshold)
        )
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    # ── Embedding ────────────────────────────────────────────────
    def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for the prompt text using Local Semantics (0 tokens)."""
        try:
            from app.local_semantics import get_embedding
            emb = get_embedding(text)
            # Convert numpy array to python list for psycopg
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        except Exception as exc:
            logger.warning("SemanticCache failed to generate embedding: %s", exc)
            return None

    def _ensure_schema(self, dimension: int) -> bool:
        """Make sure the cache table matches the local encoder's dimension.

        The table was pinned to vector(384) for all-MiniLM-L6-v2, but
        local_semantics prefers the fine-tuned BGE model (768 dims) whenever it
        is present on disk. That mismatch made every read and write raise, and
        the exception handlers turned it into a permanent silent cache miss.
        """
        if self._schema_ready:
            return True

        with self._schema_lock:
            if self._schema_ready:
                return True
            try:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                        cur.execute(
                            f"""
                            CREATE TABLE IF NOT EXISTS {CACHE_TABLE} (
                                id SERIAL PRIMARY KEY,
                                prompt_text TEXT NOT NULL,
                                prompt_embedding vector({dimension}) NOT NULL,
                                generated_response JSONB NOT NULL,
                                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                            )
                            """
                        )
                        cur.execute(
                            """
                            SELECT a.atttypmod
                            FROM pg_attribute a
                            JOIN pg_class c ON c.oid = a.attrelid
                            WHERE c.relname = %s AND a.attname = 'prompt_embedding' AND a.attnum > 0
                            """,
                            (CACHE_TABLE,),
                        )
                        row = cur.fetchone()
                        current_dimension = int(row[0]) if row and row[0] and row[0] > 0 else None

                        if current_dimension is not None and current_dimension != dimension:
                            # Cached vectors from a different encoder are not
                            # comparable to the current one, so they are worthless
                            # rather than merely stale: rebuild the table.
                            logger.warning(
                                "SemanticCache: encoder dimension changed (%s → %s), rebuilding %s",
                                current_dimension, dimension, CACHE_TABLE,
                            )
                            cur.execute(f"DROP TABLE IF EXISTS {CACHE_TABLE}")
                            cur.execute(
                                f"""
                                CREATE TABLE {CACHE_TABLE} (
                                    id SERIAL PRIMARY KEY,
                                    prompt_text TEXT NOT NULL,
                                    prompt_embedding vector({dimension}) NOT NULL,
                                    generated_response JSONB NOT NULL,
                                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                                )
                                """
                            )

                        cur.execute(
                            f"""
                            CREATE INDEX IF NOT EXISTS idx_{CACHE_TABLE}_embedding_hnsw
                            ON {CACHE_TABLE} USING hnsw (prompt_embedding vector_cosine_ops)
                            WITH (m = 16, ef_construction = 64)
                            """
                        )
                        # Exact-prompt lookups drive the upsert path below.
                        cur.execute(
                            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{CACHE_TABLE}_prompt "
                            f"ON {CACHE_TABLE} (md5(prompt_text))"
                        )
                        conn.commit()
                self._schema_ready = True
                return True
            except Exception as exc:
                logger.error("SemanticCache schema check failed: %s", exc)
                return False

    # ── Read ─────────────────────────────────────────────────────
    def search_cache(self, prompt: str) -> dict[str, Any] | None:
        """Search the semantic cache for a highly similar prompt."""
        if not settings.cache_enabled or not _get_dsn():
            return None

        embedding = self._get_embedding(prompt)
        if not embedding:
            return None
        if not self._ensure_schema(len(embedding)):
            return None

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Cosine similarity = 1 - cosine_distance, so
                    # similarity >= threshold  ⇔  distance <= 1 - threshold.
                    distance_threshold = 1.0 - self.threshold
                    embedding_str = f"[{','.join(str(x) for x in embedding)}]"

                    cur.execute(
                        f"""
                        SELECT generated_response, 1 - (prompt_embedding <=> %s::vector) AS similarity
                        FROM {CACHE_TABLE}
                        WHERE prompt_embedding <=> %s::vector <= %s
                          AND created_at > NOW() - make_interval(days => %s)
                        ORDER BY prompt_embedding <=> %s::vector
                        LIMIT 1
                        """,
                        (
                            embedding_str,
                            embedding_str,
                            distance_threshold,
                            max(1, int(settings.semantic_cache_ttl_days)),
                            embedding_str,
                        ),
                    )
                    row = cur.fetchone()

            if row:
                response_json, similarity = row
                logger.info(
                    "Semantic Cache HIT: similarity=%.3f for prompt='%s'", similarity, prompt[:50]
                )
                # JSONB comes back decoded; a TEXT column would not.
                if isinstance(response_json, str):
                    try:
                        response_json = json.loads(response_json)
                    except json.JSONDecodeError:
                        logger.warning("SemanticCache: stored response is not valid JSON, ignoring")
                        return None
                if isinstance(response_json, dict):
                    response_json["_semantic_cache_hit"] = True
                    return response_json
                logger.warning("SemanticCache: stored response is not an object, ignoring")
                return None

            logger.debug("Semantic Cache MISS for prompt='%s'", prompt[:50])
            return None

        except Exception as exc:
            logger.error("SemanticCache search failed: %s", exc)
            return None

    # ── Write ────────────────────────────────────────────────────
    def save_cache(self, prompt: str, response: dict[str, Any]) -> None:
        """Save a generated response to the semantic cache."""
        if not settings.cache_enabled or not _get_dsn():
            return

        # Clean the response from cache flags before saving
        clean_response = dict(response)
        clean_response.pop("_semantic_cache_hit", None)

        embedding = self._get_embedding(prompt)
        if not embedding:
            return
        if not self._ensure_schema(len(embedding)):
            return

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    embedding_str = f"[{','.join(str(x) for x in embedding)}]"

                    # Refresh the existing row instead of appending a duplicate
                    # every time the same prompt misses the similarity threshold.
                    cur.execute(
                        f"""
                        INSERT INTO {CACHE_TABLE} (prompt_text, prompt_embedding, generated_response)
                        VALUES (%s, %s::vector, %s)
                        ON CONFLICT (md5(prompt_text))
                        DO UPDATE SET
                            prompt_embedding = EXCLUDED.prompt_embedding,
                            generated_response = EXCLUDED.generated_response,
                            created_at = NOW()
                        """,
                        (prompt, embedding_str, json.dumps(clean_response)),
                    )
                    # Bounded retention: drop anything past the TTL.
                    cur.execute(
                        f"DELETE FROM {CACHE_TABLE} WHERE created_at < NOW() - make_interval(days => %s)",
                        (max(1, int(settings.semantic_cache_ttl_days)),),
                    )
                conn.commit()
            logger.info("Semantic Cache SAVED for prompt='%s'", prompt[:50])
        except Exception as exc:
            logger.error("SemanticCache save failed: %s", exc)


semantic_cache = PgSemanticCache()
