# Week 3 Main Document

## Summary
Week 3 moved the project from prototype quality to operational quality.

## Task Summary
- Planned tasks: 14
- Completed tasks: 14
- Completion rate: 100%

## Week Scope
Week 3 focused on operational backend readiness and data pipeline hardening.

## Main Workstreams
1. Backend hardening and runtime controls.
2. Agentic RAG runtime integration.
3. Incremental source collection and ingestion.
4. Vector ingestion with pruning.
5. Runtime observability and analytics support.

## Backend Hardening

### API and Runtime Controls
1. Added runtime-aware health endpoint.
2. Added runtime metrics endpoint.
3. Added quality gate enforcement.
4. Added in-memory and optional Redis rate limiting.

### Agentic RAG Runtime
1. Added feature toggles for live LLM, vector retrieval, and agent mode.
2. Added guarded website tool behavior by allowed domain list.
3. Added fallback behavior for reliability.

### Evidence
1. app/api_server.py
2. app/langchain_pipeline.py
3. app/config.py

## Data Pipeline

### Source Collection
1. Implemented AU source collector for HTML and PDF.
2. Added domain allowlist for controlled crawling.
3. Added incremental source-state hashing.

### Vector Ingestion
1. Added canonical JSONL to pgvector ingestion.
2. Added incremental embedding-state hashing.
3. Added pruning for deleted chunks.

### Evidence
1. app/collect_au_sources.py
2. app/ingest_pgvector.py
3. app/ingest_vector_store.py

## Benchmark and Validation

### Benchmark Coverage
1. Retrieval benchmark recorded for week-level evaluation.
2. Runtime and observability behavior validated by API-level checks.

### Validation Highlights
1. Backend tests pass after hardening changes.
2. Frontend build passes after analytics/runtime integration.

### Evidence
1. docs/week3_benchmark_latest.md
2. tests/test_api_reports.py
3. ui/angular-frontend (build output)

## Completion Checklist

### Task Count
- Planned tasks: 14
- Completed tasks: 14

### Checklist
1. Retrieval guardrails enabled.
2. Need More Context safety path implemented.
3. Chat API hardened.
4. Report persistence completed.
5. Agentic RAG integrated.
6. Live runtime feature toggles added.
7. Runtime health status exposed.
8. Runtime metrics endpoint exposed.
9. Quality gate enforcement enabled.
10. Rate limiting implemented.
11. AU source collector added.
12. Incremental source-state ingestion enabled.
13. pgvector prune behavior added.
14. Frontend analytics/runtime view completed.

### Main Evidence
1. docs/week3_plan.md
2. docs/week3_benchmark_latest.md
3. app/api_server.py
4. app/langchain_pipeline.py
5. app/collect_au_sources.py
6. app/ingest_pgvector.py

## Expected Outcome
By the end of Week 3, the system should include quality guardrails, runtime health/metrics visibility, and incremental ingestion capabilities.

