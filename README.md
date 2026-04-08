# AI Blog Generator for Backoffice

## Description
Prototype chatbot ho tro nhan vien backoffice tao ban nhap blog tu prompt text tu do bang cach ket hop LLM va RAG.

## Main Features
- Nhan prompt dang chat
- Phan tich yeu cau nguoi dung (intent, topic, tone, audience)
- Truy xuat tai lieu lien quan tu knowledge base mau
- Sinh title, outline va blog draft
- Ho tro chinh sua qua hoi thoai nhieu luot

## Folder Structure
- app/: code logic chinh
- data/: du lieu raw, processed va metadata
- ui/: giao dien demo bang Streamlit
- notebooks/: noi thu nghiem nhanh
- tests/: unit tests
- docs/: tai lieu bai toan, kien truc va ke hoach

## Quick Start
1. Cai package:

```bash
pip install -r requirements.txt
```

2. Chay CLI prototype:

```bash
python -m app.main
```

3. Chay UI demo:

```bash
streamlit run ui/streamlit_app.py
```

UI hien tai la chat-style frontend:
- Nhap prompt trong o chat ben duoi
- Test follow-up prompt nhu "Make it shorter" trong cung session
- Dung "Clear chat session" o sidebar de reset context

## Angular Frontend (Test Truc Tiep)

### 1) Chay backend API
```bash
python -m pip install -r requirements.txt
uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000
```

### 2) Chay Angular app
```bash
cd ui/angular-frontend
npm install
npm start
```

Mo trinh duyet tai http://localhost:4200.

Frontend Angular se goi API:
- GET http://localhost:8000/api/health
- POST http://localhost:8000/api/chat

Goi y test:
1. Write a blog about what a police check is for first-time job applicants, in a clear and professional tone.
2. Make it shorter

4. Chay test:

```bash
python -m pytest -q
```

5. Test parser nhanh:

```bash
python -c "from app.parser import parse_user_input; print(parse_user_input('Write a blog about police checks for job seekers'))"
```

6. Test retrieval nhanh:

```bash
python -c "from app.retriever import retrieve_top_k; print([d.doc_id for d in retrieve_top_k('What documents are required?', 'data/processed', 3)])"
```

## Current Output Schema (Generator)
Generator output duoc chuan hoa theo format:

```json
{
	"title": "...",
	"outline": ["...", "..."],
	"draft": "...",
	"sources_used": ["doc_id_1", "doc_id_2"]
}
```

Flow CLI hien tai in day du 4 phan:
- Parsed result
- Retrieved top docs
- Generated title/outline/draft
- Sources used

## Cap Nhat Dataset cho RAG

Khi them file moi vao data/raw, chay ingest de cap nhat data/processed va metadata:

```bash
python -m app.ingest_data
```

Ingest script se:
- Lam sach text co ban
- Dong bo file sang data/processed
- Upsert metadata vao data/metadata/documents.json

## Guardrails Retrieval

Retriever da co:
- Domain guard: query ngoai pham vi dataset se tra ve "Need More Context"
- Confidence guard: top score/thong tin retrieval thap se khong tao draft de tranh hallucination

Ban co the chinh threshold bang env vars:
- MIN_TOP_SCORE (mac dinh 3)
- MIN_CONFIDENCE (mac dinh 0.35)

## Week 3: Hybrid Retrieval + Benchmark

Retriever hien tai dung hybrid scoring:
- Lexical overlap
- Metadata rerank (topic/document_type/approved)
- Concept semantic similarity

Chay benchmark tu dong:

```bash
python -m app.evaluate_benchmark
```

Report duoc tao tai:
- docs/week3_benchmark_latest.md

## Scope
Project nay la prototype doc lap trong pham vi thuc tap, khong can thiep source code chinh cua he thong hien tai.

## Core Modules
- app/parser.py: parse prompt text tu do
- app/retriever.py: retrieve top tai lieu lien quan tu data mau
- app/generator.py: tao output chuan (title, outline, draft, sources_used)
- app/main.py: demo CLI full flow parser -> retriever -> generator
