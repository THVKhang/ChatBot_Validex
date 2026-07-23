"""Shared PostgreSQL connection pool.

Provides a singleton ``ConnectionPool`` backed by ``psycopg_pool``.
All modules that need a database connection should use
:func:`get_connection` instead of calling ``psycopg.connect()`` directly.

Graceful fallback: if the pool cannot be created (e.g. no DATABASE_URL or
psycopg_pool not installed), :func:`get_connection` falls back to a direct
``psycopg.connect()`` call so the system remains functional on a single-server
development setup.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────
_DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
_DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "20"))
_DB_POOL_TIMEOUT = float(os.getenv("DB_POOL_TIMEOUT", "10.0"))

# ── Singleton Pool ───────────────────────────────────────
_pool = None
_pool_lock = threading.Lock()
_pool_init_attempted = False


def _get_dsn() -> str:
    """Resolve the database connection string."""
    dsn = os.getenv("DATABASE_URL", "").strip()
    if dsn:
        return dsn
    alt = os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
    if alt.startswith("postgresql+psycopg2://"):
        return "postgresql://" + alt.split("postgresql+psycopg2://", 1)[1]
    return alt


def _init_pool() -> Any | None:
    """Create the connection pool (called once on first use)."""
    global _pool, _pool_init_attempted
    if _pool_init_attempted:
        return _pool

    dsn = _get_dsn()
    if not dsn:
        logger.info("db_pool: No DATABASE_URL configured — pool disabled, using direct connections.")
        _pool_init_attempted = True
        return None

    try:
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=max(1, _DB_POOL_MIN),
            max_size=max(2, _DB_POOL_MAX),
            timeout=_DB_POOL_TIMEOUT,
            # Automatically reconnect broken connections
            check=ConnectionPool.check_connection,
            open=True,
        )
        logger.info(
            "db_pool: Connection pool created (min=%d, max=%d, timeout=%.1fs)",
            _DB_POOL_MIN, _DB_POOL_MAX, _DB_POOL_TIMEOUT,
        )
        _pool_init_attempted = True
        return _pool
    except ImportError:
        logger.warning("db_pool: psycopg_pool not installed — falling back to direct connections.")
    except Exception as exc:
        logger.warning("db_pool: Failed to create pool (%s) — falling back to direct connections.", exc)

    _pool_init_attempted = True
    return None


def get_pool():
    """Return the singleton pool instance (or None if unavailable)."""
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        return _init_pool()


@contextmanager
def get_connection() -> Generator:
    """Borrow a connection from the pool (preferred) or open a direct one.

    Usage::

        from app.db_pool import get_connection

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    pool = get_pool()
    conn = None
    from_pool = False

    if pool is not None:
        try:
            # Only catch errors during connection *acquisition*, not during use
            conn = pool.getconn()
            from_pool = True
        except Exception as exc:
            logger.warning("db_pool: Pool getconn failed (%s), trying direct connect.", exc)

    if conn is None:
        # Fallback: direct connection
        dsn = _get_dsn()
        if not dsn:
            raise RuntimeError("No DATABASE_URL configured and connection pool unavailable.")
        import psycopg
        conn = psycopg.connect(dsn)

    try:
        yield conn
    finally:
        if conn is not None:
            if from_pool and pool is not None:
                try:
                    import psycopg
                    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                        conn.rollback()
                except Exception:
                    pass
                try:
                    pool.putconn(conn)
                except Exception:
                    pass
            elif not from_pool:
                try:
                    conn.close()
                except Exception:
                    pass


def pool_stats() -> dict[str, Any]:
    """Return connection pool statistics for the health endpoint."""
    pool = get_pool()
    if pool is None:
        return {"status": "disabled", "reason": "no pool configured"}

    try:
        stats = pool.get_stats()
        return {
            "status": "active",
            "pool_min": pool.min_size,
            "pool_max": pool.max_size,
            "pool_size": stats.get("pool_size", 0),
            "pool_available": stats.get("pool_available", 0),
            "requests_waiting": stats.get("requests_waiting", 0),
            "requests_num": stats.get("requests_num", 0),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def check_db_connectivity() -> dict[str, Any]:
    """Ping the database to verify connectivity."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                return {"status": "ok", "result": result[0] if result else None}
    except RuntimeError:
        return {"status": "no_database_url"}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def close_pool() -> None:
    """Close the pool on application shutdown."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        try:
            logger.info("db_pool: Connection pool closed.")
        except (ValueError, OSError):
            pass  # stdout may already be closed during atexit
        _pool = None


# Ensure pool is closed on process exit (prevents pytest cleanup warnings)
import atexit
atexit.register(close_pool)
