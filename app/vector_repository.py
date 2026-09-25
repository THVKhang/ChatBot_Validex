from abc import ABC, abstractmethod
from typing import Any, List, Dict
import json
import logging
import re
from app.config import settings
from app.database import DatabaseManager

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

# Table names are interpolated into SQL (identifiers cannot be bound as
# parameters). config._normalize_pgvector_table already enforces this shape on
# the configured value; re-checking here protects callers that pass one in.
_SAFE_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _safe_table(table_name: str) -> str:
    """Validate a table name before it is interpolated into a statement."""
    if not _SAFE_TABLE_NAME_RE.match(str(table_name or "")):
        raise ValueError(
            f"Unsafe table name {table_name!r}: expected letters, digits and underscores only."
        )
    return table_name


def build_keyword_tsquery(query: str) -> str:
    """Turn a topic string into a websearch_to_tsquery expression that can match.

    websearch_to_tsquery ANDs bare terms together, so a normal multi-word topic
    ("spent convictions scheme Crimes Act") requires every term to appear in the
    same chunk and matches nothing — which silently reduced the hybrid search to
    its semantic half. ORing the terms lets the keyword branch contribute, and
    ts_rank still ranks chunks covering more of the terms higher.
    """
    terms = [term for term in _WORD_RE.findall(query) if len(term) > 2]
    return " OR ".join(terms) if terms else query


class VectorStoreRepository(ABC):
    """Abstract interface for Vector Store interactions."""

    @abstractmethod
    def initialize_schema(self, table_name: str, dimension: int) -> None:
        """Create extensions, target tables, columns, and indexes if they do not exist."""
        pass

    @abstractmethod
    def upsert_records(self, table_name: str, records: List[Dict[str, Any]]) -> None:
        """Upsert records (containing metadata, text, and vector embeddings) into the vector table."""
        pass

    @abstractmethod
    def delete_records(self, table_name: str, chunk_ids: List[str]) -> int:
        """Delete records from the vector table by chunk IDs. Returns deleted count."""
        pass

    @abstractmethod
    def hybrid_search(
        self,
        table_name: str,
        query: str,
        query_vector: List[float],
        top_k: int,
        require_non_fake: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search (similarity + keyword search) combining them using Reciprocal Rank Fusion (RRF)."""
        pass

    @abstractmethod
    def upsert_parents(self, table_name: str, parents: List[Dict[str, Any]]) -> None:
        """Upsert parent documents into the parents table."""
        pass

    @abstractmethod
    def get_parent_content(self, table_name: str, parent_id: str) -> str | None:
        """Retrieve full parent text by parent_id."""
        pass


class PGVectorRepository(VectorStoreRepository):
    """PostgreSQL PGVector database implementation of the VectorStoreRepository interface."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()

    def table_embedding_dimension(self, table_name: str) -> int | None:
        """Return the declared vector dimension of ``table_name.embedding``.

        Returns ``None`` when the table (or the column) does not exist yet.
        """
        table_name = _safe_table(table_name)
        with self.db_manager.get_connection(prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.atttypmod
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    WHERE c.relname = %s AND a.attname = 'embedding' AND a.attnum > 0
                    """,
                    (table_name,),
                )
                row = cur.fetchone()
        if not row or row[0] is None or row[0] <= 0:
            return None
        return int(row[0])

    def stored_embedding_providers(self, table_name: str) -> set[str]:
        """Return the distinct embedding providers already present in the table.

        Empty when the table does not exist yet.
        """
        table_name = _safe_table(table_name)
        with self.db_manager.get_connection(prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass(%s)", (table_name,))
                row = cur.fetchone()
                if not row or row[0] is None:
                    return set()
                cur.execute(
                    f"SELECT DISTINCT coalesce(embedding_provider, 'unknown') FROM {table_name}"
                )
                return {str(r[0]) for r in cur.fetchall()}

    def initialize_schema(self, table_name: str, dimension: int) -> None:
        table_name = _safe_table(table_name)
        # Guard against silently writing into a table built for a different
        # embedding model: pgvector would reject the INSERT with an opaque
        # error, and 'CREATE TABLE IF NOT EXISTS' below would not fix it.
        existing_dimension = self.table_embedding_dimension(table_name)
        if existing_dimension is not None and existing_dimension != dimension:
            raise RuntimeError(
                f"Table '{table_name}' stores vector({existing_dimension}) embeddings but the "
                f"configured embedding model produces {dimension} dimensions. Re-ingest into a "
                f"new table, or run app/refresh_embeddings.py to rebuild the existing one."
            )

        with self.db_manager.get_connection(prepare_threshold=None) as conn:
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
                        embedding vector({dimension}) NOT NULL,
                        status TEXT NOT NULL DEFAULT 'in_force',
                        jurisdiction TEXT NOT NULL DEFAULT 'Commonwealth',
                        document_type TEXT NOT NULL DEFAULT 'webpage',
                        act_name TEXT NOT NULL DEFAULT '',
                        section_ref TEXT NOT NULL DEFAULT '',
                        effective_date TEXT NOT NULL DEFAULT '',
                        parent_context TEXT NOT NULL DEFAULT '',
                        parent_id TEXT,
                        last_verified_at TIMESTAMPTZ,
                        superseded_by TEXT DEFAULT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        fts_content tsvector GENERATED ALWAYS AS (
                            to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
                        ) STORED
                    )
                    """
                )
                # Create the parents table to hold the full parent chunks
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name}_parents (
                        parent_id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        doc_id TEXT NOT NULL,
                        metadata TEXT NOT NULL DEFAULT '{{}}'
                    )
                    """
                )
                # Safe migration for existing tables
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS chunk_hash TEXT")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS embedding_provider TEXT")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'in_force'")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS jurisdiction TEXT NOT NULL DEFAULT 'Commonwealth'")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS document_type TEXT NOT NULL DEFAULT 'webpage'")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS act_name TEXT NOT NULL DEFAULT ''")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS section_ref TEXT NOT NULL DEFAULT ''")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS effective_date TEXT NOT NULL DEFAULT ''")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS parent_context TEXT NOT NULL DEFAULT ''")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS parent_id TEXT")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMPTZ")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS superseded_by TEXT")
                # hybrid_search() reads fts_content; without it every keyword branch
                # raises and the whole hybrid search silently degrades to the local
                # file fallback. Keep this in sync with sql/database.sql.
                cur.execute(
                    f"""
                    ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS fts_content tsvector
                    GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))
                    ) STORED
                    """
                )
                # Indexes
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_topic ON {table_name}(topic)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_domain ON {table_name}(source_domain)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_status ON {table_name}(status)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_parent_id ON {table_name}(parent_id)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_jurisdiction ON {table_name}(jurisdiction)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_status_jurisdiction ON {table_name}(status, jurisdiction)")
                # Keyword branch of hybrid_search
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_fts ON {table_name} USING GIN (fts_content)"
                )
                # ANN index for the semantic branch. Without it pgvector falls back
                # to a sequential scan over the whole table on every query.
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_hnsw
                    ON {table_name} USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """
                )
                conn.commit()

    def upsert_records(self, table_name: str, records: List[Dict[str, Any]]) -> None:
        table_name = _safe_table(table_name)
        if not records:
            return

        # Ensure parent_id is present in all records (even if None) to avoid key errors in cursor
        for r in records:
            if "parent_id" not in r:
                r["parent_id"] = None

        upsert_sql = f"""
            INSERT INTO {table_name} (
                chunk_id, chunk_hash, embedding_provider,
                doc_id, source_url, source_domain, source_type,
                topic, region, title, authority_score, approved,
                content, embedding,
                status, jurisdiction, document_type, act_name,
                section_ref, effective_date, parent_context, parent_id
            ) VALUES (
                %(chunk_id)s, %(chunk_hash)s, %(embedding_provider)s,
                %(doc_id)s, %(source_url)s, %(source_domain)s, %(source_type)s,
                %(topic)s, %(region)s, %(title)s, %(authority_score)s, %(approved)s,
                %(content)s, %(embedding)s::vector,
                %(status)s, %(jurisdiction)s, %(document_type)s, %(act_name)s,
                %(section_ref)s, %(effective_date)s, %(parent_context)s, %(parent_id)s
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
                embedding = EXCLUDED.embedding,
                status = EXCLUDED.status,
                jurisdiction = EXCLUDED.jurisdiction,
                document_type = EXCLUDED.document_type,
                act_name = EXCLUDED.act_name,
                section_ref = EXCLUDED.section_ref,
                effective_date = EXCLUDED.effective_date,
                parent_context = EXCLUDED.parent_context,
                parent_id = EXCLUDED.parent_id
        """

        with self.db_manager.get_connection(prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.executemany(upsert_sql, records)
                conn.commit()

    def delete_records(self, table_name: str, chunk_ids: List[str]) -> int:
        table_name = _safe_table(table_name)
        if not chunk_ids:
            return 0
        with self.db_manager.get_connection(prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {table_name} WHERE chunk_id = ANY(%s)",
                    (chunk_ids,),
                )
                deleted_count = cur.rowcount or 0
                conn.commit()
                return deleted_count

    def hybrid_search(
        self,
        table_name: str,
        query: str,
        query_vector: List[float],
        top_k: int,
        require_non_fake: bool = False,
        status_filter: str = "in_force",
        jurisdiction_filter: str | None = None,
        min_similarity: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic + keyword with RRF fusion.

        The two branches are fused with a FULL OUTER JOIN on chunk_id so a chunk
        found by *both* branches accumulates both reciprocal-rank terms — that is
        the entire point of RRF, and a UNION ALL cannot express it (it emits the
        same chunk twice, each row carrying only half the score).

        CRITICAL: Defaults to status='in_force' to prevent citing repealed legislation.
        """
        table_name = _safe_table(table_name)
        vector_literal = "[" + ",".join(f"{float(item):.8f}" for item in query_vector) + "]"

        if min_similarity is None:
            min_similarity = float(getattr(settings, "pgvector_min_similarity", 0.0) or 0.0)

        params: Dict[str, Any] = {
            "query_vector": vector_literal,
            "query": build_keyword_tsquery(query),
            "pool": max(1, top_k) * 4,
            "top_k": top_k,
            "min_similarity": min_similarity,
        }

        # Filters are parameterised: these values reach the query as bind
        # parameters, never as interpolated SQL text.
        extra_filters = ""
        if require_non_fake:
            extra_filters += " AND coalesce(embedding_provider, 'unknown') != 'fake'"
        if status_filter:
            extra_filters += " AND status = %(status_filter)s"
            params["status_filter"] = status_filter
        if jurisdiction_filter:
            extra_filters += " AND jurisdiction = %(jurisdiction_filter)s"
            params["jurisdiction_filter"] = jurisdiction_filter

        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH semantic_pool AS (
                        -- Pure ANN top-N so the HNSW index can serve the ordering;
                        -- ranking happens afterwards over just these rows.
                        SELECT
                            chunk_id,
                            1 - (embedding <=> %(query_vector)s::vector) AS semantic_similarity
                        FROM {table_name}
                        WHERE approved = true {extra_filters}
                        ORDER BY embedding <=> %(query_vector)s::vector
                        LIMIT %(pool)s
                    ),
                    semantic_search AS (
                        SELECT
                            chunk_id,
                            semantic_similarity,
                            ROW_NUMBER() OVER (ORDER BY semantic_similarity DESC) AS semantic_rank
                        FROM semantic_pool
                    ),
                    keyword_pool AS (
                        SELECT
                            chunk_id,
                            ts_rank(fts_content, websearch_to_tsquery('english', %(query)s)) AS keyword_score
                        FROM {table_name}
                        WHERE approved = true {extra_filters}
                          AND fts_content @@ websearch_to_tsquery('english', %(query)s)
                        ORDER BY keyword_score DESC
                        LIMIT %(pool)s
                    ),
                    keyword_search AS (
                        SELECT
                            chunk_id,
                            keyword_score,
                            ROW_NUMBER() OVER (ORDER BY keyword_score DESC) AS keyword_rank
                        FROM keyword_pool
                    ),
                    fused AS (
                        SELECT
                            COALESCE(s.chunk_id, k.chunk_id) AS chunk_id,
                            s.semantic_similarity,
                            k.keyword_score,
                            s.semantic_rank,
                            k.keyword_rank,
                            COALESCE(1.0 / (60 + s.semantic_rank), 0.0)
                              + COALESCE(1.0 / (60 + k.keyword_rank), 0.0) AS rrf_score
                        FROM semantic_search s
                        FULL OUTER JOIN keyword_search k ON s.chunk_id = k.chunk_id
                    )
                    SELECT
                        t.chunk_id, t.doc_id, t.content, t.source_url, t.source_domain, t.source_type,
                        t.topic, t.region, t.title, t.authority_score, t.approved,
                        t.jurisdiction, t.act_name, t.section_ref, t.parent_context, t.status, t.parent_id,
                        t.effective_date, t.last_verified_at, t.created_at,
                        COALESCE(f.semantic_similarity, 0.0) AS similarity,
                        f.keyword_score,
                        f.semantic_rank, f.keyword_rank, f.rrf_score
                    FROM fused f
                    JOIN {table_name} t ON t.chunk_id = f.chunk_id
                    -- Relevance floor: a chunk pulled in by the ANN branch must clear
                    -- min_similarity. Keyword-only hits have no cosine and pass through.
                    WHERE f.semantic_similarity IS NULL
                       OR f.semantic_similarity >= %(min_similarity)s
                    ORDER BY f.rrf_score DESC
                    LIMIT %(top_k)s;
                    """,
                    params,
                )
                rows = cur.fetchall()

        results = []
        for row in rows:
            results.append({
                "chunk_id": row[0],
                "doc_id": row[1],
                "content": row[2],
                "source_url": row[3],
                "source_domain": row[4],
                "source_type": row[5],
                "topic": row[6],
                "region": row[7],
                "title": row[8],
                "authority_score": row[9],
                "approved": row[10],
                "jurisdiction": row[11],
                "act_name": row[12],
                "section_ref": row[13],
                "parent_context": row[14],
                "status": row[15],
                "parent_id": row[16],
                "effective_date": row[17],
                "last_verified_at": row[18],
                "created_at": row[19],
                "similarity": row[20],
                "keyword_score": row[21],
                "semantic_rank": row[22],
                "keyword_rank": row[23],
                "rrf_score": row[24],
            })
        return results

    def upsert_parents(self, table_name: str, parents: List[Dict[str, Any]]) -> None:
        table_name = _safe_table(table_name)
        if not parents:
            return
        upsert_sql = f"""
            INSERT INTO {table_name}_parents (
                parent_id, content, doc_id, metadata
            ) VALUES (
                %(parent_id)s, %(content)s, %(doc_id)s, %(metadata)s
            )
            ON CONFLICT (parent_id)
            DO UPDATE SET
                content = EXCLUDED.content,
                doc_id = EXCLUDED.doc_id,
                metadata = EXCLUDED.metadata
        """
        with self.db_manager.get_connection(prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.executemany(upsert_sql, parents)
                conn.commit()

    def get_parent_content(self, table_name: str, parent_id: str) -> str | None:
        table_name = _safe_table(table_name)
        sql = f"SELECT content FROM {table_name}_parents WHERE parent_id = %s"
        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (parent_id,))
                row = cur.fetchone()
                return row[0] if row else None

    def mark_repealed(
        self,
        table_name: str,
        chunk_ids: List[str],
        superseded_by: str = "",
    ) -> int:
        """Soft-delete: mark chunks as repealed instead of deleting.

        Used by Delta Sync when legislation is amended/repealed.
        Old chunks are kept for audit trail but excluded from search.
        """
        table_name = _safe_table(table_name)
        if not chunk_ids:
            return 0
        with self.db_manager.get_connection(prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE {table_name} SET status = 'repealed', superseded_by = %s WHERE chunk_id = ANY(%s)",
                    (superseded_by, chunk_ids),
                )
                count = cur.rowcount or 0
                conn.commit()
                return count
