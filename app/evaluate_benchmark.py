from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.retriever import retrieve_with_guard


def run_benchmark(
    benchmark_path: str = "docs/benchmark_week3.json",
    report_path: str = "docs/week3_benchmark_latest.md",
) -> dict:
    cases = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    rows: list[dict] = []

    pass_count = 0
    for case in cases:
        query = case["query"]
        decision = retrieve_with_guard(
            query,
            settings.data_processed_dir,
            settings.top_k,
            settings.metadata_path,
            settings.min_top_score,
            settings.min_confidence,
        )

        top_doc = decision.docs[0].doc_id if decision.docs else "none"
        expected_top = case.get("expected_top_doc")
        expected_status = case.get("expected_status", "ok")

        is_pass = True
        if expected_top and top_doc != expected_top:
            is_pass = False
        if decision.status != expected_status:
            is_pass = False

        if is_pass:
            pass_count += 1

        rows.append(
            {
                "query": query,
                "expected_status": expected_status,
                "actual_status": decision.status,
                "expected_top_doc": expected_top or "-",
                "actual_top_doc": top_doc,
                "confidence": round(decision.confidence, 3),
                "pass": is_pass,
            }
        )

    total = len(rows)
    pass_rate = round((pass_count / total) * 100, 2) if total else 0.0

    lines = [
        "# Week 3 Benchmark Report",
        "",
        f"- Total cases: {total}",
        f"- Passed: {pass_count}",
        f"- Pass rate: {pass_rate}%",
        "",
        "| Query | Expected Status | Actual Status | Expected Top | Actual Top | Confidence | Result |",
        "|---|---|---|---|---|---:|---|",
    ]

    for item in rows:
        lines.append(
            f"| {item['query']} | {item['expected_status']} | {item['actual_status']} | "
            f"{item['expected_top_doc']} | {item['actual_top_doc']} | {item['confidence']} | "
            f"{'PASS' if item['pass'] else 'FAIL'} |"
        )

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return {"total": total, "passed": pass_count, "pass_rate": pass_rate}


if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps(result, indent=2))
