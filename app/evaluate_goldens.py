from __future__ import annotations

import json
from pathlib import Path

from app.main import process_prompt
from app.session_manager import SessionManager
from app.utils import tokenize


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    overlap = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(overlap) / max(1, len(union))


def run_golden_evaluation(
    benchmark_path: str = "data/benchmark/generation_queries.json",
    goldens_path: str = "data/goldens/golden_answers.json",
    report_path: str = "data/benchmark/golden_report.md",
    min_similarity: float = 0.18,
) -> dict:
    benchmark_cases = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    golden_map_raw = json.loads(Path(goldens_path).read_text(encoding="utf-8"))
    golden_map = {item["id"]: item for item in golden_map_raw}

    session = SessionManager()
    rows: list[dict] = []
    pass_count = 0

    for case in benchmark_cases:
        prompt = case["prompt"]
        golden_id = case["golden_id"]
        golden = golden_map.get(golden_id)

        payload = process_prompt(prompt, session)
        generated = payload.get("generated", {})
        generated_title = str(generated.get("title", ""))
        generated_draft = str(generated.get("draft", ""))

        if not golden:
            similarity = 0.0
            is_pass = False
            golden_title = "missing_golden"
            golden_excerpt = ""
        else:
            golden_title = str(golden.get("title", ""))
            golden_text = str(golden.get("golden_draft", ""))
            similarity = _jaccard_similarity(generated_draft, golden_text)
            is_pass = similarity >= min_similarity
            golden_excerpt = golden_text[:180].replace("\n", " ")

        if is_pass:
            pass_count += 1

        rows.append(
            {
                "prompt": prompt,
                "golden_id": golden_id,
                "golden_title": golden_title,
                "generated_title": generated_title,
                "similarity": round(similarity, 3),
                "result": "PASS" if is_pass else "FAIL",
                "golden_excerpt": golden_excerpt,
            }
        )

    total = len(rows)
    pass_rate = round((pass_count / total) * 100, 2) if total else 0.0

    lines = [
        "# Golden Evaluation Report",
        "",
        f"- Total cases: {total}",
        f"- Passed: {pass_count}",
        f"- Pass rate: {pass_rate}%",
        f"- Min similarity threshold: {min_similarity}",
        "",
        "| Prompt | Golden ID | Golden Title | Generated Title | Similarity | Result |",
        "|---|---|---|---|---:|---|",
    ]

    for item in rows:
        lines.append(
            f"| {item['prompt']} | {item['golden_id']} | {item['golden_title']} | "
            f"{item['generated_title']} | {item['similarity']} | {item['result']} |"
        )

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return {
        "total": total,
        "passed": pass_count,
        "pass_rate": pass_rate,
        "threshold": min_similarity,
    }


if __name__ == "__main__":
    result = run_golden_evaluation()
    print(json.dumps(result, indent=2))
