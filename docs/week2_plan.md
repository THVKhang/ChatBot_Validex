# Week 2 Completion Checklist

## Muc tieu Week 2
Den cuoi tuan 2, prototype can demo duoc flow:
User prompt -> Parser -> Retriever -> Generator -> Output

## Pham vi Week 2
- Tap trung nang chat luong parser, retrieval, generation
- On dinh flow end-to-end
- Chua can UI dep
- Chua can multi-turn chat hoan chinh

## Bang Done / Not done / Evidence
| Hang muc | Dieu kien dat | Trang thai | Evidence |
|---|---|---|---|
| 1. Parser nang cap | Nhan intent: create_blog / rewrite / shorten | [x] | app/parser.py, tests/test_parser.py |
| 2. Parser default rules | Co default audience, tone, length, intent | [x] | app/parser.py, tests/test_parser.py |
| 3. Du lieu processed sach hon | Da review chunking, bo chunk rac, metadata day du | [x] | data/processed/, data/metadata/documents.json, docs/metadata_schema.md |
| 4. Retriever nang cap | Tra top-k hop ly hon Week 1 | [x] | app/retriever.py, docs/retrieval_testcases.md |
| 5. Retrieval testcases | Co 5-10 query test va danh gia | [x] | docs/retrieval_testcases.md |
| 6. Generator prototype | Sinh title + outline + draft ngan | [x] | app/generator.py, tests/test_generator.py |
| 7. Output schema chuan | Output co title, outline, draft, sources_used | [x] | app/generator.py, app/main.py |
| 8. End-to-end flow | Chay prompt -> parse -> retrieve -> generate | [x] | app/main.py, ui/streamlit_app.py |
| 9. README cap nhat | Co huong dan parser/retriever/full demo | [x] | README.md |
| 10. Demo case mentor | Co 1 case demo dep + output luu lai | [x] | docs/week2_demo_case.md |

## Parser target output
```json
{
  "intent": "create_blog",
  "topic": "police checks",
  "audience": "job seekers",
  "tone": "professional",
  "length": "medium"
}
```

## Parser default rules
- audience mac dinh: general audience
- tone mac dinh: clear, professional
- length mac dinh: medium
- intent mac dinh: create_blog

## Metadata schema can co
- id
- title
- source
- topic
- document_type
- approved
- content

## Query test goi y cho retrieval
1. What is a police check?
2. How long does a police check take?
3. Who needs a police check?
4. What documents are required?
5. Police check for employment

## Ke hoach theo ngay
- Ngay 1: review Week 1, nang parser fields
- Ngay 2: them default rules, test parser voi 5-10 prompt
- Ngay 3: chinh data processed, metadata
- Ngay 4: nang retriever, test top-k
- Ngay 5: hoan tat retrieval testcases
- Ngay 6: lam generator prototype
- Ngay 7: noi flow full, cap nhat README, chuan bi demo

## Dau ra bat buoc cuoi Week 2
- Parser tot hon Week 1
- Retriever tot hon Week 1
- Generator prototype
- Flow parse -> retrieve -> generate chay duoc
- It nhat 1 demo case hoan chinh

## Tu danh gia Week 2 (5 cau hoi)
1. Prompt tu do vao, parser co hieu co ban khong?
2. Retriever co lay tai lieu lien quan hon Week 1 khong?
3. Generator co sinh duoc outline hoac draft khong?
4. Flow end-to-end co chay on dinh khong?
5. Co 1 demo case dep de bao mentor khong?

## Cau bao cao ngan Week 2
Trong tuan 2, em tap trung hoan thien parser cho input chat tu do, cai thien chat luong retrieval thong qua chuan hoa du lieu va bo test query, dong thoi phat trien generator prototype de noi thanh flow hoan chinh tu prompt den blog draft co ban.

## Evidence update (Day 1-2)
- Parser da nang cap intent + default rules trong app/parser.py.
- Da them 10 prompt test parser va default tone test trong tests/test_parser.py.
- Ket qua test: 17 passed (python -m pytest -q).
- Da chay 10 retrieval queries va ghi ket qua trong docs/retrieval_testcases.md.

## Evidence update (Day 3-4)
- Da bo sung du lieu processed: doc_07_required_documents, doc_08_who_needs_police_check.
- Da cap nhat metadata voi fields document_type va approved trong data/metadata/documents.json.
- Da nang cap retriever scoring theo intent query + synonym expansion trong app/retriever.py.
- Da them retriever tests cho processing-time va required-documents query trong tests/test_retriever.py.
- Ket qua test moi: 19 passed (python -m pytest -q).
- Retrieval benchmark cai thien: 9/10 pass (truoc do 6/10).

## Evidence update (Day 5-7)
- Da nang cap generator output schema: title, outline, draft, sources_used.
- Da noi full flow trong app/main.py voi output Parsed -> Retrieved -> Generated.
- Da cap nhat UI demo de hien thi parsed result, retrieved docs, generated output.
- Da them test cho schema generator; tong test hien tai: 20 passed.
- Da cap nhat README voi huong dan parser/retriever/full flow.
- Da dien week2_demo_case.md bang output thuc te tu demo prompt.

## Evidence update (Day 8)
- Da them topic post-processing trong app/parser.py de bo clause du thua ve tone/audience.
- Da them title formatter trong app/generator.py de tao title gon gang hon.
- Ket qua demo prompt: topic duoc clean tu cau dai thanh "what a police check is".
- Da bo sung test parser/generator cho day 8; tong test hien tai: 24 passed.

## Evidence update (Day 9)
- Da bo sung quality controls cho generator theo intent (create/rewrite/shorten).
- Da bo sung quality controls cho do dai draft theo length (short/medium/long).
- Da them test day 9 trong tests/test_generator.py; tong test hien tai: 27 passed.
- Kiem chung thuc te: short draft ~766 ky tu, long draft ~1673 ky tu tren cung chu de.

## Evidence update (Day 10)
- Da nang cap retriever voi metadata-aware rerank (topic/document_type/approved).
- Da bo sung mapping file_stem trong data/metadata/documents.json.
- Ket qua benchmark: fix query compliance, dat 10/10 pass tren bo 10 query.

## Evidence update (Day 11)
- Da nang cap session_manager luu parsed_intent, parsed_topic, generated_draft.
- Da noi context reuse cho prompt rewrite/shorten trong app/main.py.
- Kiem chung luot 2 "Make it shorter" su dung bai vua tao (context: using previous draft).

## Evidence update (Day 12)
- Da chot evaluation snapshot day 12 voi so lieu truoc/sau.
- File tong hop: docs/evaluation_snapshot_day12.md.
- Tong test hien tai: 29 passed.

## Evidence update (Day 13-14)
- Da tao bo demo 5 phut voi 2 use case trong docs/demo_script.md.
- Da hoan tat week report/handoff trong docs/week2_report.md.
- Da danh dau Day 13-14 DONE trong docs/day8_to_day14_plan.md.
