from app.main import process_prompt
from app.session_manager import SessionManager
from app.evaluate_benchmark import run_benchmark


def test_process_prompt_includes_retrieval_meta_and_snippet():
    payload = process_prompt("Write a blog about police check", SessionManager())
    assert "retrieval_meta" in payload
    assert "status" in payload["retrieval_meta"]
    if payload["retrieved"]:
        assert "snippet" in payload["retrieved"][0]


def test_run_benchmark_generates_report(tmp_path):
    benchmark_file = tmp_path / "bench.json"
    report_file = tmp_path / "report.md"
    benchmark_file.write_text(
        """
[
  {"query": "What is a police check?", "expected_status": "ok"},
  {"query": "Write a travel blog about Vietnam", "expected_status": "out_of_domain"}
]
        """.strip(),
        encoding="utf-8",
    )

    result = run_benchmark(str(benchmark_file), str(report_file))
    assert result["total"] == 2
    assert report_file.exists()
