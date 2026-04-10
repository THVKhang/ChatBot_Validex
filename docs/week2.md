# Week 2 Main Document

## Summary
Week 2 focused on quality improvements and stable end-to-end demo behavior.

## Task Summary
- Planned tasks: 12
- Completed tasks: 12
- Completion rate: 100%

## Week Scope
Week 2 improved quality and stability of the baseline pipeline.

## Main Workstreams
1. Parser intent and defaults improvements.
2. Retrieval quality and reranking improvements.
3. Generator schema and content control improvements.
4. Session-aware follow-up behavior.
5. Demo packaging and reporting.

## Parser and Retrieval Improvements

### Parser Enhancements
1. Added intent support: create_blog, rewrite, shorten.
2. Added default field rules for audience, tone, and length.
3. Added topic cleanup for noisy prompts.

### Retrieval Enhancements
1. Improved scoring behavior for better top-k ranking.
2. Added metadata-aware reranking.
3. Expanded retrieval test cases.

### Evidence
1. app/parser.py
2. app/retriever.py
3. tests/test_parser.py
4. docs/retrieval_testcases.md

## Generation and Session

### Generator Improvements
1. Standardized output schema: title, outline, draft, sources_used.
2. Added intent-based behavior controls.
3. Added length-based draft controls.

### Session Improvements
1. Added previous-turn context for rewrite and shorten prompts.
2. Improved continuity in follow-up prompts.

### Evidence
1. app/generator.py
2. app/main.py
3. app/session_manager.py
4. tests/test_generator.py
5. tests/test_main_flow.py

## Demo and Reporting

### Demo Packaging
1. Prepared a structured demo case for mentor walkthrough.
2. Added a reusable demo script for consistent presentation.

### Reporting
1. Captured the week summary and constraints.
2. Recorded evidence and outcomes.

### Evidence
1. docs/week2_demo_case.md
2. docs/demo_script.md
3. docs/week2_report.md
4. docs/evaluation_snapshot_day12.md

## Completion Checklist

### Task Count
- Planned tasks: 12
- Completed tasks: 12

### Checklist
1. Parser intents expanded.
2. Parser defaults added.
3. Topic cleaner implemented.
4. Processed dataset improved.
5. Retriever ranking improved.
6. Metadata-aware rerank added.
7. Retrieval test cases documented.
8. Generator schema standardized.
9. Generator quality controls implemented.
10. Session-aware follow-up behavior added.
11. End-to-end flow stabilized.
12. Demo and report finalized.

### Main Evidence
1. docs/week2_plan.md
2. docs/week2_report.md
3. docs/retrieval_testcases.md

## Expected Outcome
By the end of Week 2, the system should provide stable end-to-end output with clearer parser behavior and better retrieval relevance.
