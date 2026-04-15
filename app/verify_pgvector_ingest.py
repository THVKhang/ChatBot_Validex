from __future__ import annotations

import json
import os

import psycopg
from psycopg import sql
from dotenv import load_dotenv


load_dotenv()


def _database_url() -> str:
    mode = os.getenv("INGEST_MODE", "raw_sql").strip().lower()
    if mode == "langchain":
        value = os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
        if value.startswith("postgresql+psycopg2://"):
            return "postgresql://" + value.split("postgresql+psycopg2://", 1)[1]
        return value
    return os.getenv("DATABASE_URL", "").strip()


def verify_ingest(table_name: str = "validex_knowledge") -> dict:
    conn_str = _database_url()
    if not conn_str:
        return {
            "status": "error",
            "message": "Missing database connection string",
        }

    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("select extname from pg_extension where extname = 'vector'")
                vector_enabled = cur.fetchone() is not None

                cur.execute(
                    "select to_regclass(%s)",
                    (table_name,),
                )
                table_exists = cur.fetchone()[0] is not None
                if not table_exists:
                    return {
                        "status": "error",
                        "message": f"Table {table_name} not found",
                        "vector_extension_enabled": vector_enabled,
                    }

                cur.execute(sql.SQL("select count(*) from {} ").format(sql.Identifier(table_name)))
                total_rows = cur.fetchone()[0]

                cur.execute(
                    sql.SQL(
                        """
                        select count(distinct source_domain), count(distinct source_type), count(distinct topic)
                        from {}
                        """
                    ).format(sql.Identifier(table_name))
                )
                domains_count, source_types_count, topics_count = cur.fetchone()

                cur.execute(
                    sql.SQL(
                        """
                        select coalesce(min(authority_score), 0), coalesce(max(authority_score), 0)
                        from {}
                        """
                    ).format(sql.Identifier(table_name))
                )
                min_auth, max_auth = cur.fetchone()

                cur.execute(
                    """
                    select indexname
                    from pg_indexes
                    where tablename = %s
                    order by indexname
                    """,
                    (table_name,),
                )
                indexes = [row[0] for row in cur.fetchall()]

                cur.execute(
                    sql.SQL(
                        """
                        select chunk_id, (embedding <=> embedding) as self_distance
                        from {}
                        limit 3
                        """
                    ).format(sql.Identifier(table_name))
                )
                vector_sanity = [
                    {"chunk_id": row[0], "self_distance": float(row[1])}
                    for row in cur.fetchall()
                ]
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

    return {
        "status": "ok",
        "table": table_name,
        "vector_extension_enabled": vector_enabled,
        "rows": total_rows,
        "distinct": {
            "source_domains": domains_count,
            "source_types": source_types_count,
            "topics": topics_count,
        },
        "authority_score_range": {
            "min": float(min_auth),
            "max": float(max_auth),
        },
        "indexes": indexes,
        "vector_sanity": vector_sanity,
    }


if __name__ == "__main__":
    print(json.dumps(verify_ingest(), indent=2))
