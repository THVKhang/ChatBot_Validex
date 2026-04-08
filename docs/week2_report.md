# Week 2 Report

## Tong quan
Trong Week 2, em tap trung nang cap chat luong parser, retrieval, va generator de bien bo khung Week 1 thanh prototype co the demo end-to-end on dinh.

## Ket qua chinh
1. Parser
- Bo sung intent: create_blog / rewrite / shorten
- Bo sung default rules cho audience, tone, length
- Them topic cleaner de bo clause du thua

2. Retrieval
- Nang cap scoring + synonym expansion
- Them metadata-aware rerank theo topic/document_type/approved
- Benchmark cai thien tu 6/10 len 10/10 pass

3. Generator
- Chuan hoa output schema: title, outline, draft, sources_used
- Them quality controls theo length (short/medium/long)
- Them quality controls theo intent (create/rewrite/shorten)

4. End-to-end flow
- CLI output day du Parsed -> Retrieved -> Generated -> Sources
- Ho tro pseudo multi-turn: "Make it shorter" dung context draft truoc

## So lieu xac thuc
- Test suite: 29 passed
- Retrieval benchmark: 10/10 pass
- Multi-turn check: da co "context: using previous draft"

## Minh chung chinh
- app/parser.py
- app/retriever.py
- app/generator.py
- app/main.py
- app/session_manager.py
- docs/retrieval_testcases.md
- docs/evaluation_snapshot_day12.md
- docs/week2_demo_case.md

## Han che hien tai
- Chua co semantic retrieval (hien tai lexical + heuristic + metadata boost)
- Chua co production concerns: auth, moderation, logging/monitoring, deployment
- UI chua day du chat multi-turn nhu san pham that

## Backlog de xuat (Week tiep theo)
1. Them semantic retrieval (embedding + similarity)
2. Viet prompt template chi tiet hon cho tung intent
3. Bo sung evaluator cho quality generation (rubric scoring)
4. Hoan thien UI chat multi-turn + history view
5. Chuan bi deployment demo (local server + basic observability)

## Ket luan
Week 2 dat muc "prototype dung duoc" voi flow end-to-end on dinh, co benchmark retrieval, va co kha nang xu ly follow-up shorten/rewrite o muc co ban.
