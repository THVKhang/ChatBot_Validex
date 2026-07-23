"""Thread-safe metrics collector.

Replaces the plain ``dict`` used in ``api_server._metrics`` with a class
that uses ``threading.Lock`` to prevent race conditions under concurrent
request handling.

Drop-in replacement: import ``metrics`` singleton and call
``metrics.increment("chat_requests_total")`` instead of
``_metrics["chat_requests_total"] += 1``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

from app.config import settings


class ThreadSafeMetrics:
    """Atomic counters and sliding-window latency tracker."""

    def __init__(self, window_size: int = 200) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {
            "chat_requests_total": 0,
            "chat_errors_total": 0,
            "quality_gate_blocked_total": 0,
        }
        self._distributions: dict[str, dict[str, int]] = {
            "generation_mode_count": defaultdict(int),
            "retrieval_mode_count": defaultdict(int),
        }
        self._latency_samples: deque[float] = deque(maxlen=max(20, window_size))

    # ── Counters ─────────────────────────────────────────
    def increment(self, counter_name: str, amount: int = 1) -> None:
        """Atomically increment a named counter."""
        with self._lock:
            self._counters[counter_name] = self._counters.get(counter_name, 0) + amount

    def get_counter(self, counter_name: str) -> int:
        """Return current value of a counter."""
        with self._lock:
            return self._counters.get(counter_name, 0)

    # ── Distributions ────────────────────────────────────
    def record_distribution(self, distribution_name: str, key: str) -> None:
        """Atomically increment a distribution bucket."""
        with self._lock:
            dist = self._distributions.setdefault(distribution_name, defaultdict(int))
            dist[key] += 1

    # ── Latency ──────────────────────────────────────────
    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample in the sliding window."""
        with self._lock:
            self._latency_samples.append(latency_ms)

    # ── Snapshot ─────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        """Return a consistent snapshot of all metrics."""
        with self._lock:
            samples = list(self._latency_samples)
            counters = dict(self._counters)
            distributions = {
                name: dict(dist)
                for name, dist in self._distributions.items()
            }

        avg_latency = round(sum(samples) / len(samples), 2) if samples else 0.0
        p95_latency = 0.0
        if samples:
            ordered = sorted(samples)
            index = int(0.95 * (len(ordered) - 1))
            p95_latency = round(float(ordered[index]), 2)

        return {
            **counters,
            **distributions,
            "latency": {
                "samples": len(samples),
                "avg_ms": avg_latency,
                "p95_ms": p95_latency,
            },
        }

    # ── Prometheus ───────────────────────────────────────
    def prometheus_text(self) -> str:
        """Generate Prometheus-compatible text exposition."""
        from prometheus_client import CollectorRegistry, Counter, generate_latest

        registry = CollectorRegistry()
        snap = self.snapshot()

        req_counter = Counter("validex_chat_requests_total", "Total chat requests", registry=registry)
        err_counter = Counter("validex_chat_errors_total", "Total chat errors", registry=registry)
        qg_counter = Counter("validex_quality_gate_blocks_total", "Total quality gate blocks", registry=registry)

        req_counter.inc(snap.get("chat_requests_total", 0))
        err_counter.inc(snap.get("chat_errors_total", 0))
        qg_counter.inc(snap.get("quality_gate_blocked_total", 0))

        return generate_latest(registry).decode("utf-8")


# ── Singleton ────────────────────────────────────────────
metrics = ThreadSafeMetrics(window_size=settings.metrics_window_size)
