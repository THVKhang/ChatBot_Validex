"""CLI entry point for Knowledge Gap Analysis & Data Sourcing.

Usage:
    python run_gap_analysis.py --sweep           # Proactive Prompt Sweeping
    python run_gap_analysis.py --ml-rejections   # ML Gate Rejection Analysis
    python run_gap_analysis.py --matrix          # State-by-State Coverage Matrix
    python run_gap_analysis.py --all             # Run all 3 tools
    python run_gap_analysis.py --crawl           # Crawl golden sources
    python run_gap_analysis.py --crawl --ingest  # Crawl + auto-ingest into pgvector
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validex Knowledge Gap Analysis & Data Sourcing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sweep", action="store_true",
        help="Run Proactive Prompt Sweeping (vector-only, no LLM cost)",
    )
    parser.add_argument(
        "--ml-rejections", action="store_true",
        help="Analyze ML Gate rejections to find data-hungry topics",
    )
    parser.add_argument(
        "--matrix", action="store_true",
        help="Generate State-by-State Coverage Matrix",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all 3 analysis tools",
    )
    parser.add_argument(
        "--crawl", action="store_true",
        help="Crawl golden sources (.gov.au)",
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Auto-ingest crawled data into pgvector (use with --crawl)",
    )
    parser.add_argument(
        "--collection", type=str, default="validex_knowledge_v2",
        help="Target collection for ingestion (default: validex_knowledge_v2)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Similarity threshold for blind spot detection (default: 0.65)",
    )

    args = parser.parse_args()

    if not any([args.sweep, args.ml_rejections, args.matrix, args.all, args.crawl]):
        parser.print_help()
        sys.exit(1)

    # ── Analysis Tools ──
    if args.all:
        from app.knowledge_gap_analyzer import run_all
        run_all()

    else:
        if args.sweep:
            from app.knowledge_gap_analyzer import run_prompt_sweep
            run_prompt_sweep(threshold=args.threshold)

        if args.ml_rejections:
            from app.knowledge_gap_analyzer import run_ml_rejection_analysis
            run_ml_rejection_analysis()

        if args.matrix:
            from app.knowledge_gap_analyzer import run_coverage_matrix
            run_coverage_matrix()

    # ── Crawler ──
    if args.crawl:
        from app.gov_crawler import crawl_golden_sources, CRAWLED_OUTPUT_PATH
        result = crawl_golden_sources()

        if result.get("success", 0) > 0 and args.ingest:
            print(f"\n[Ingest] Auto-ingesting crawled data into collection: {args.collection}")
            from app.ingest_pgvector import ingest_jsonl_to_pgvector
            ingest_result = ingest_jsonl_to_pgvector(
                jsonl_path=CRAWLED_OUTPUT_PATH,
                table_name=args.collection,
            )
            print(json.dumps(ingest_result, indent=2))


if __name__ == "__main__":
    main()
