"""Knowledge Gap Analyzer — Detect blind spots in Validex's knowledge base.

Three analysis tools:
1. Proactive Prompt Sweeping: Run edge-case topics through vector search (no LLM cost)
2. ML Gate Rejection Analyzer: Surface topics the system is "data-hungry" for
3. State-by-State Coverage Matrix: Map [Topic × State] to find empty cells

Usage:
    from app.knowledge_gap_analyzer import run_prompt_sweep, run_ml_rejection_analysis, run_coverage_matrix
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────
EDGE_CASE_TOPICS_PATH = "data/benchmark/edge_case_topics.json"
TRAINING_DATA_PATH = "data/ml/training_data.jsonl"
CANONICAL_CHUNKS_PATH = "data/canonical/au_blog_chunks.jsonl"
SIMILARITY_THRESHOLD = 0.65
REPORT_OUTPUT_DIR = "data/reports"

# ─── Canonical topic/state labels ─────────────────────────────────────
TOPICS = [
    "police_check", "wwcc", "ndis", "spent_convictions",
    "100_point_id", "aged_care", "healthcare", "education",
    "employer_compliance", "immigration", "security_clearance",
]

STATES = ["Commonwealth", "NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]


# ═══════════════════════════════════════════════════════════════════════
# TOOL 1: Proactive Prompt Sweeping
# ═══════════════════════════════════════════════════════════════════════

def run_prompt_sweep(
    topics_path: str = EDGE_CASE_TOPICS_PATH,
    threshold: float = SIMILARITY_THRESHOLD,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run vector-only retrieval for edge-case topics. No LLM cost.

    Returns a report dict with blind_spots, covered_topics, and full results.
    """
    from app.config import settings
    from app.langchain_pipeline import pipeline

    topics = _load_topics(topics_path)
    if not topics:
        return {"error": "No topics found", "path": topics_path}

    dsn = pipeline._pgvector_connection_dsn()
    if not dsn:
        return {"error": "No DATABASE_URL configured for pgvector"}

    logger.info("Prompt Sweep: scanning %d topics (threshold=%.2f)", len(topics), threshold)

    results: list[dict] = []
    blind_spots: list[dict] = []
    covered: list[dict] = []

    for i, topic in enumerate(topics):
        query_vector = pipeline._query_embedding(topic)
        if not query_vector:
            results.append({"topic": topic, "error": "embedding failed"})
            continue

        try:
            from app.vector_repository import PGVectorRepository
            repo = PGVectorRepository()
            rows = repo.hybrid_search(
                table_name=settings.pgvector_table,
                query=topic,
                query_vector=query_vector,
                top_k=top_k,
                require_non_fake=False,
            )
        except Exception as exc:
            results.append({"topic": topic, "error": str(exc)})
            continue

        top_scores = []
        top_titles = []
        for row in rows[:3]:
            sim = float(row.get("similarity", 0.0))
            title = str(row.get("title", ""))
            top_scores.append(round(sim, 4))
            top_titles.append(title)

        # Take the BEST semantic similarity, not the first row's. Rows come back
        # ordered by RRF, and a keyword-only hit carries no cosine score (0.0),
        # so reading position 0 reported a well-covered topic as a blind spot
        # whenever the keyword branch happened to win the fusion.
        best_sim = max(top_scores) if top_scores else 0.0
        is_blind_spot = best_sim < threshold

        entry = {
            "topic": topic,
            "top_1_similarity": best_sim,
            "top_3_similarities": top_scores,
            "top_3_titles": top_titles,
            "is_blind_spot": is_blind_spot,
            "num_results": len(rows),
        }
        results.append(entry)

        if is_blind_spot:
            blind_spots.append(entry)
        else:
            covered.append(entry)

        if (i + 1) % 10 == 0:
            logger.info("Prompt Sweep: %d/%d topics scanned", i + 1, len(topics))

    report = {
        "total_topics": len(topics),
        "covered_count": len(covered),
        "blind_spot_count": len(blind_spots),
        "coverage_pct": round(len(covered) / max(len(topics), 1) * 100, 1),
        "threshold": threshold,
        "blind_spots": blind_spots,
        "covered": covered,
        "all_results": results,
    }

    _save_report(report, "prompt_sweep_report.json")
    _print_sweep_summary(report)
    return report


def _load_topics(path: str) -> list[str]:
    """Load edge-case topics from JSON file."""
    p = Path(path)
    if not p.exists():
        logger.error("Topics file not found: %s", path)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(t).strip() for t in data if str(t).strip()]
    except Exception as exc:
        logger.error("Failed to parse topics file: %s", exc)
    return []


def _print_sweep_summary(report: dict) -> None:
    """Print a human-readable summary of the sweep results."""
    print("\n" + "=" * 70)
    print("[Sweep] PROACTIVE PROMPT SWEEP REPORT")
    print("=" * 70)
    print(f"  Topics scanned:    {report['total_topics']}")
    print(f"  Coverage:          {report['coverage_pct']}%")
    print(f"  Covered topics:    {report['covered_count']}")
    print(f"  [Blind spots]:    {report['blind_spot_count']}")
    print(f"  Threshold:         {report['threshold']}")

    if report["blind_spots"]:
        print(f"\n{'-' * 70}")
        print("BLIND SPOTS (Top-1 Similarity < threshold):")
        print(f"{'-' * 70}")
        for bs in sorted(report["blind_spots"], key=lambda x: x["top_1_similarity"]):
            sim_str = f"{bs['top_1_similarity']:.3f}"
            print(f"  [{sim_str}] {bs['topic'][:70]}")
            if bs["top_3_titles"]:
                print(f"           Best match: {bs['top_3_titles'][0][:60]}")

    print(f"\n{'-' * 70}")
    print(f"Report saved to: {REPORT_OUTPUT_DIR}/prompt_sweep_report.json")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# TOOL 2: ML Gate Rejection Analyzer
# ═══════════════════════════════════════════════════════════════════════

def run_ml_rejection_analysis(
    data_path: str = TRAINING_DATA_PATH,
) -> dict[str, Any]:
    """Analyze ML Gate rejections to find data-hungry topics.

    Reads training_data.jsonl and surfaces topics that were rejected
    most frequently, with their rejection reasons.
    """
    p = Path(data_path)
    if not p.exists():
        return {"error": f"Training data not found: {data_path}"}

    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not records:
        return {"error": "No training records found"}

    # Filter for rejected samples
    rejected = [
        r for r in records
        if r.get("labels", {}).get("quality_class") == "low"
    ]

    ml_gate_rejected = [
        r for r in rejected
        if r.get("labels", {}).get("label_source") == "ml_gate_reject"
    ]

    # Analyze rejection topics
    topic_counts: dict[str, int] = defaultdict(int)
    reason_counts: dict[str, int] = defaultdict(int)
    topic_reasons: dict[str, list[str]] = defaultdict(list)
    high_nli_topics: list[dict] = []

    for r in rejected:
        topic = r.get("metadata", {}).get("topic", "unknown")
        reason = r.get("metadata", {}).get("rejection_reason", "unknown")
        nli = r.get("features", {}).get("nli_contradictions", 0)

        topic_counts[topic] += 1
        reason_counts[reason] += 1
        if topic not in topic_reasons or len(topic_reasons[topic]) < 3:
            topic_reasons[topic].append(reason)

        if nli and nli > 0.3:
            high_nli_topics.append({
                "topic": topic,
                "nli_contradictions": round(nli, 3),
                "reason": reason,
            })

    report = {
        "total_training_records": len(records),
        "total_rejected": len(rejected),
        "ml_gate_rejected": len(ml_gate_rejected),
        "rejection_rate_pct": round(len(rejected) / max(len(records), 1) * 100, 1),
        "top_rejected_topics": dict(sorted(topic_counts.items(), key=lambda x: -x[1])[:15]),
        "top_rejection_reasons": dict(sorted(reason_counts.items(), key=lambda x: -x[1])[:10]),
        "topic_reason_samples": dict(topic_reasons),
        "high_nli_contradiction_topics": high_nli_topics[:20],
    }

    _save_report(report, "ml_rejection_report.json")
    _print_rejection_summary(report)
    return report


def _print_rejection_summary(report: dict) -> None:
    """Print a human-readable summary of ML Gate rejections."""
    print("\n" + "=" * 70)
    print("[ML Gate] ML GATE REJECTION ANALYSIS")
    print("=" * 70)
    print(f"  Total training records:  {report['total_training_records']}")
    print(f"  Total rejected:          {report['total_rejected']}")
    print(f"  ML Gate rejected:        {report['ml_gate_rejected']}")
    print(f"  Rejection rate:          {report['rejection_rate_pct']}%")

    if report["top_rejected_topics"]:
        print(f"\n{'-' * 70}")
        print("MOST REJECTED TOPICS (data-hungry areas):")
        for topic, count in report["top_rejected_topics"].items():
            bar = "#" * min(count, 30)
            print(f"  {topic[:40]:<40} {count:>3} {bar}")

    if report["top_rejection_reasons"]:
        print(f"\n{'-' * 70}")
        print("TOP REJECTION REASONS:")
        for reason, count in report["top_rejection_reasons"].items():
            print(f"  [{count:>3}] {reason[:65]}")

    if report["high_nli_contradiction_topics"]:
        print(f"\n{'-' * 70}")
        print("[Warning] HIGH NLI CONTRADICTION TOPICS (LLM hallucinating):")
        for item in report["high_nli_contradiction_topics"][:10]:
            print(f"  NLI={item['nli_contradictions']:.3f} | {item['topic'][:50]}")

    print(f"\n{'-' * 70}")
    print(f"Report saved to: {REPORT_OUTPUT_DIR}/ml_rejection_report.json")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════
# TOOL 3: State-by-State Coverage Matrix
# ═══════════════════════════════════════════════════════════════════════

def run_coverage_matrix(
    chunks_path: str = CANONICAL_CHUNKS_PATH,
) -> dict[str, Any]:
    """Build a [Topic × State] coverage matrix from the canonical JSONL.

    Highlights empty cells as critical blind spots.
    """
    p = Path(chunks_path)
    if not p.exists():
        return {"error": f"Chunks file not found: {chunks_path}"}

    records = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # Build matrix
    matrix: dict[str, dict[str, int]] = {
        topic: {state: 0 for state in STATES} for topic in TOPICS
    }

    # Map chunks to topic/state categories
    topic_keyword_map = {
        "police_check": ["police check", "police clearance", "criminal history", "npcs", "criminal record check"],
        "wwcc": ["working with children", "wwcc", "blue card", "vulnerable people"],
        "ndis": ["ndis", "disability", "worker screening"],
        "spent_convictions": ["spent conviction", "rehabilitation", "annulled", "expunged", "waiting period"],
        "100_point_id": ["100 point", "identity verification", "identity check", "identity document", "austrac"],
        "aged_care": ["aged care", "elderly", "residential care"],
        "healthcare": ["ahpra", "health practitioner", "nurse", "medical", "pharmacy", "dental"],
        "education": ["teacher", "education", "school", "early childhood"],
        "employer_compliance": ["employer", "privacy", "fair work", "pre-employment", "hiring"],
        "immigration": ["visa", "immigration", "vevo", "right to work", "migrant"],
        "security_clearance": ["security clearance", "agsva", "nv1", "nv2", "baseline"],
    }

    unmatched_count = 0
    for rec in records:
        text = (str(rec.get("text", "")) + " " + str(rec.get("title", ""))).lower()
        jurisdiction = str(rec.get("jurisdiction", "")).strip()

        # Normalize jurisdiction
        state = _normalize_jurisdiction(jurisdiction)

        # Match topic
        matched_topic = rec.get("topic")
        if matched_topic not in TOPICS:
            matched_topic = None
            for topic, keywords in topic_keyword_map.items():
                if any(kw in text for kw in keywords):
                    matched_topic = topic
                    break

        if matched_topic and state in STATES:
            matrix[matched_topic][state] += 1
        elif matched_topic:
            # Has topic but unrecognized state — count under Commonwealth
            matrix[matched_topic]["Commonwealth"] += 1
        else:
            unmatched_count += 1

    # Find empty cells (blind spots)
    blind_spots = []
    for topic in TOPICS:
        for state in STATES:
            if matrix[topic][state] == 0:
                blind_spots.append({"topic": topic, "state": state})

    report = {
        "total_chunks": len(records),
        "unmatched_chunks": unmatched_count,
        "matrix": matrix,
        "total_cells": len(TOPICS) * len(STATES),
        "empty_cells": len(blind_spots),
        "coverage_pct": round((1 - len(blind_spots) / (len(TOPICS) * len(STATES))) * 100, 1),
        "blind_spots": blind_spots,
    }

    _save_report(report, "coverage_matrix_report.json")
    _print_coverage_matrix(report)
    return report


def _normalize_jurisdiction(jurisdiction: str) -> str:
    """Normalize jurisdiction strings to canonical state codes."""
    j = jurisdiction.strip().upper()
    mapping = {
        "COMMONWEALTH": "Commonwealth",
        "CTH": "Commonwealth",
        "FEDERAL": "Commonwealth",
        "NEW SOUTH WALES": "NSW",
        "VICTORIA": "VIC",
        "QUEENSLAND": "QLD",
        "WESTERN AUSTRALIA": "WA",
        "SOUTH AUSTRALIA": "SA",
        "TASMANIA": "TAS",
        "AUSTRALIAN CAPITAL TERRITORY": "ACT",
        "NORTHERN TERRITORY": "NT",
    }
    if j in mapping:
        return mapping[j]
    # Try direct match
    if jurisdiction.strip() in STATES:
        return jurisdiction.strip()
    return "Commonwealth"  # Default fallback


def _print_coverage_matrix(report: dict) -> None:
    """Print the coverage matrix as a formatted table."""
    matrix = report["matrix"]

    print("\n" + "=" * 100)
    print("[Matrix] STATE-BY-STATE COVERAGE MATRIX")
    print("=" * 100)
    print(f"  Total chunks: {report['total_chunks']}  |  "
          f"Coverage: {report['coverage_pct']}%  |  "
          f"Empty cells: {report['empty_cells']}/{report['total_cells']}")
    print()

    # Header row
    header = f"{'Topic':<22}"
    for state in STATES:
        header += f"{state:>6}"
    print(header)
    print("-" * (22 + 6 * len(STATES)))

    # Data rows
    for topic in TOPICS:
        row = f"{topic:<22}"
        for state in STATES:
            count = matrix[topic][state]
            if count == 0:
                row += f"{'   X':>6}"
            else:
                row += f"{count:>6}"
        print(row)

    print("-" * (22 + 6 * len(STATES)))

    # Highlight critical blind spots
    if report["blind_spots"]:
        critical = [bs for bs in report["blind_spots"]
                    if bs["topic"] in ("police_check", "wwcc", "spent_convictions", "100_point_id")]
        if critical:
            print(f"\n[Critical Blind Spots] ({len(critical)}):")
            for bs in critical:
                print(f"  [X] [{bs['topic']}] x [{bs['state']}]")

    print(f"\nReport saved to: {REPORT_OUTPUT_DIR}/coverage_matrix_report.json")
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _save_report(report: dict, filename: str) -> None:
    """Save report to JSON file."""
    out_dir = Path(REPORT_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Report saved to %s", out_path)


def run_all() -> dict[str, Any]:
    """Run all 3 analysis tools and produce a combined report."""
    print("\n🔍 Running Knowledge Gap Analysis (all 3 tools)...\n")

    sweep_report = run_prompt_sweep()
    rejection_report = run_ml_rejection_analysis()
    matrix_report = run_coverage_matrix()

    combined = {
        "prompt_sweep": sweep_report,
        "ml_rejections": rejection_report,
        "coverage_matrix": matrix_report,
        "summary": {
            "sweep_blind_spots": sweep_report.get("blind_spot_count", 0),
            "sweep_coverage_pct": sweep_report.get("coverage_pct", 0),
            "rejection_rate_pct": rejection_report.get("rejection_rate_pct", 0),
            "matrix_coverage_pct": matrix_report.get("coverage_pct", 0),
            "matrix_empty_cells": matrix_report.get("empty_cells", 0),
        },
    }

    _save_report(combined, "knowledge_gap_full_report.json")

    print("\n" + "=" * 70)
    print("[Summary] COMBINED KNOWLEDGE GAP SUMMARY")
    print("=" * 70)
    s = combined["summary"]
    print(f"  Prompt Sweep Coverage:   {s['sweep_coverage_pct']}% "
          f"({s['sweep_blind_spots']} blind spots)")
    print(f"  ML Rejection Rate:       {s['rejection_rate_pct']}%")
    print(f"  State Matrix Coverage:   {s['matrix_coverage_pct']}% "
          f"({s['matrix_empty_cells']} empty cells)")
    print("=" * 70)

    return combined
