"""LRU response cache with TTL for the generation pipeline.

Caches generated blog outputs keyed by a hash of (topic, intent, length)
to avoid redundant LLM calls for identical queries. Entries expire after
a configurable TTL.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class ResponseCache:
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(
        self,
        max_entries: int | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        self._max_entries = max_entries or max(1, settings.cache_max_entries)
        self._ttl = ttl_seconds or max(30, settings.cache_ttl_seconds)
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def make_key(topic: str, intent: str, length: str) -> str:
        """Create a deterministic cache key from query parameters."""
        raw = f"{topic.strip().lower()}|{intent.strip().lower()}|{length.strip().lower()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def get(self, key: str) -> Any | None:
        """Return cached value if present and not expired, else None."""
        if not settings.cache_enabled:
            return None

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            stored_at, value = entry
            if time.time() - stored_at > self._ttl:
                # Expired — remove and return miss
                del self._store[key]
                self._misses += 1
                logger.debug("cache.expired key=%s", key)
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.info("cache.hit key=%s", key)
            return value

    def put(self, key: str, value: Any) -> None:
        """Store a value, evicting the oldest entry if at capacity."""
        if not settings.cache_enabled:
            return

        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = (time.time(), value)
                return

            if len(self._store) >= self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("cache.evict key=%s", evicted_key)

            self._store[key] = (time.time(), value)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._store.clear()
            logger.info("cache.cleared")

    def stats(self) -> dict[str, int]:
        """Return cache hit/miss statistics."""
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._store),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl,
            }


# Module-level singleton
response_cache = ResponseCache()
