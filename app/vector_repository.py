from abc import ABC, abstractmethod
from typing import Any, List, Dict
import json
import logging
from app.database import DatabaseManager

logger = logging.getLogger(__name__)

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


class PGVectorRepository(VectorStoreRepository):
    """PostgreSQL PGVector database implementation of the VectorStoreRepository interface."""

    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()

    def initialize_schema(self, table_name: str, dimension: int) -> None:
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
                        last_verified_at TIMESTAMPTZ,
                        superseded_by TEXT DEFAULT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMPTZ")
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS superseded_by TEXT")
                # Indexes
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_topic ON {table_name}(topic)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_source_domain ON {table_name}(source_domain)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_status ON {table_name}(status)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_jurisdiction ON {table_name}(jurisdiction)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_status_jurisdiction ON {table_name}(status, jurisdiction)")
                conn.commit()

    def upsert_records(self, table_name: str, records: List[Dict[str, Any]]) -> None:
        if not records:
            return

        upsert_sql = f"""
            INSERT INTO {table_name} (
                chunk_id, chunk_hash, embedding_provider,
                doc_id, source_url, source_domain, source_type,
                topic, region, title, authority_score, approved,
                content, embedding,
                status, jurisdiction, document_type, act_name,
                section_ref, effective_date, parent_context
            ) VALUES (
                %(chunk_id)s, %(chunk_hash)s, %(embedding_provider)s,
                %(doc_id)s, %(source_url)s, %(source_domain)s, %(source_type)s,
                %(topic)s, %(region)s, %(title)s, %(authority_score)s, %(approved)s,
                %(content)s, %(embedding)s::vector,
                %(status)s, %(jurisdiction)s, %(document_type)s, %(act_name)s,
                %(section_ref)s, %(effective_date)s, %(parent_context)s
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
                parent_context = EXCLUDED.parent_context
        """

        with self.db_manager.get_connection(prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.executemany(upsert_sql, records)
                conn.commit()

    def delete_records(self, table_name: str, chunk_ids: List[str]) -> int:
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
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic + keyword with RRF fusion.

        CRITICAL: Defaults to status='in_force' to prevent citing repealed legislation.
        """
        vector_literal = "[" + ",".join(f"{float(item):.8f}" for item in query_vector) + "]"

        extra_filters = ""
        if require_non_fake:
            extra_filters += " AND coalesce(embedding_provider, 'unknown') != 'fake'"
        if status_filter:
            extra_filters += f" AND status = '{status_filter}'"
        if jurisdiction_filter:
            extra_filters += f" AND jurisdiction = '{jurisdiction_filter}'"

        with self.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    WITH semantic_search AS (
                        SELECT 
                            chunk_id, 
                            doc_id, content, source_url, source_domain, source_type, 
                            topic, region, title, authority_score, approved,
                            jurisdiction, act_name, section_ref, parent_context, status,
                            1 - (embedding <=> %s::vector) AS similarity,
                            RANK() OVER (ORDER BY embedding <=> %s::vector) AS semantic_rank
                        FROM {table_name}
                        WHERE approved = true {extra_filters}
                        ORDER BY semantic_rank
                        LIMIT %s
                    ),
                    keyword_search AS (
                        SELECT 
                            chunk_id, 
                            doc_id, content, source_url, source_domain, source_type, 
                            topic, region, title, authority_score, approved,
                            jurisdiction, act_name, section_ref, parent_context, status,
                            ts_rank(fts_content, websearch_to_tsquery('english', %s)) AS similarity,
                            RANK() OVER (ORDER BY ts_rank(fts_content, websearch_to_tsquery('english', %s)) DESC) AS keyword_rank
                        FROM {table_name}
                        WHERE approved = true {extra_filters}
                          AND fts_content @@ websearch_to_tsquery('english', %s)
                        ORDER BY keyword_rank
                        LIMIT %s
                    )
                    SELECT 
                        chunk_id, doc_id, content, source_url, source_domain, source_type, 
                        topic, region, title, authority_score, approved,
                        jurisdiction, act_name, section_ref, parent_context, status,
                        similarity, semantic_rank, keyword_rank,
                        COALESCE(1.0 / (60 + semantic_rank), 0.0) + COALESCE(1.0 / (60 + keyword_rank), 0.0) AS rrf_score
                    FROM (
                        SELECT chunk_id, doc_id, content, source_url, source_domain, source_type, topic, region, title, authority_score, approved, jurisdiction, act_name, section_ref, parent_context, status, similarity, NULL::int AS semantic_rank, keyword_rank FROM keyword_search
                        UNION ALL
                        SELECT chunk_id, doc_id, content, source_url, source_domain, source_type, topic, region, title, authority_score, approved, jurisdiction, act_name, section_ref, parent_context, status, similarity, semantic_rank, NULL::int AS keyword_rank FROM semantic_search
                    ) combined
                    ORDER BY rrf_score DESC
                    LIMIT %s;
                    """,
                    (vector_literal, vector_literal, top_k * 2, query, query, query, top_k * 2, top_k),
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
                "similarity": row[16],
                "semantic_rank": row[17],
                "keyword_rank": row[18],
                "rrf_score": row[19],
            })
        return results

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
