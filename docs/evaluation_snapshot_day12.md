# Day 12 Evaluation Snapshot

## Scope
Tong hop ket qua sau Day 10-12 de bao cao mentor: retrieval rerank, session-aware flow, va do on dinh test.

## 1) Retrieval benchmark trend
- Baseline sau Day 4: 9/10 pass
- Sau Day 10 metadata rerank: 10/10 pass
- Bo query benchmark: docs/retrieval_testcases.md

## 2) Multi-turn behavior (Day 11)
Scenario:
1. Prompt 1: Write a blog about police check for first-time applicants
2. Prompt 2: Make it shorter

Observed:
- Prompt 2 duoc map vao topic turn truoc
- Output co dong: context: using previous draft
- Draft shorten co tham chieu "Ngu canh tu draft truoc"

Evidence:
- app/main.py
- app/session_manager.py
- tests/test_main_flow.py

## 3) Test status
- Tong test: 29 passed
- Nhom cover chinh:
  - Parser rules + topic cleaner
  - Retriever scoring + metadata rerank
  - Generator output schema + length/intent controls
  - End-to-end session-aware flow

## 4) Known limits
- Tie-break khi nhieu doc cung score van co the dao thu tu
- Chua co semantic retrieval (hien tai lexical + heuristic + metadata boost)
- Chua co UI chat multi-turn day du (moi o muc prototype)

## 5) Decision for next steps
- Day 13: dong goi demo script 5 phut + 2 use case chuan
- Day 14: week report va backlog week tiep theo
