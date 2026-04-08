# Day 8 to Day 14 Execution Plan

## Muc tieu
Nang prototype tu muc "chay duoc" len muc "demo thuyet phuc": output gon gang hon, co kiem soat quality, va co kha nang mo rong cho UI chat.

## Day 8 - Topic and title cleaner
- Tao post-processor cho topic (loai bo cum du thua nhu "in a clear and professional tone")
- Tao ham title formatter ngan gon
- Dau ra: title de doc hon trong generator
- Evidence: app/parser.py, app/generator.py, tests/test_parser.py

Status: DONE
- Prompt demo da duoc clean topic: "what a police check is"
- Test suite hien tai: 24 passed

## Day 9 - Generator quality controls
- Them draft length mode (short/medium/long) ro rang hon
- Them toc do tao outline theo intent (create/rewrite/shorten)
- Dau ra: draft on dinh hon theo length
- Evidence: app/generator.py, tests/test_generator.py

Status: DONE
- Generator da co control theo intent: create/rewrite/shorten
- Generator da co control theo length: short/medium/long
- Test suite: 27 passed

## Day 10 - Retrieval reranking pass
- Bo sung rerank theo metadata topic + document_type
- Them tie-break cho query time/documents/compliance
- Dau ra: tang pass retrieval benchmark
- Evidence: app/retriever.py, docs/retrieval_testcases.md

Status: DONE
- Da them rerank theo metadata topic/document_type/approved
- Query fail compliance da duoc fix
- Benchmark retrieval dat 10/10

## Day 11 - Session-aware rewrite
- Su dung session_manager de ho tro prompt kieu "make it shorter" / "rewrite this"
- Neu intent la shorten/rewrite thi uu tien bai vua sinh o turn truoc
- Dau ra: demo pseudo multi-turn co gia tri
- Evidence: app/session_manager.py, app/main.py, tests/

Status: DONE
- Prompt shorten/rewrite da su dung context tu turn truoc
- Da bo sung test main flow cho multi-turn context

## Day 12 - Evaluation snapshot
- Chot bo query benchmark retrieval va bo prompt benchmark parser/generator
- Ghi ket qua truoc/sau vao bang
- Dau ra: co so lieu de bao mentor
- Evidence: docs/retrieval_testcases.md, docs/week2_demo_case.md

Status: DONE
- Da tao docs/evaluation_snapshot_day12.md
- Da cap nhat bang retrieval testcases voi so lieu moi nhat

## Day 13 - Demo packaging
- Tao file demo script 5 phut (cac lenh + ky vong output)
- Chuan bi 2 use cases: create blog, rewrite/shorten
- Dau ra: thao tac demo nhat quan
- Evidence: docs/demo_script.md

Status: DONE
- Da tao docs/demo_script.md voi kich ban 5 phut
- Da co 2 use case ro rang: create blog va shorten follow-up

## Day 14 - Weekly report and handoff
- Tong ket da lam duoc, han che hien tai, backlog week tiep theo
- Chot "done/not done/evidence" cho ca 14 ngay
- Dau ra: bao cao gui mentor
- Evidence: docs/week2_report.md

Status: DONE
- Da tao docs/week2_report.md
- Da tong hop ket qua, han che, va backlog week tiep theo

## Definition of Done (Day 14)
- Retrieval benchmark >= 9/10 pass duy tri on dinh
- Generator output schema co title/outline/draft/sources_used
- Demo 2 case chay duoc (create + rewrite/shorten)
- README va docs cap nhat day du
