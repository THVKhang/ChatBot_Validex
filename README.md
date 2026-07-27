# ChatBot Validex

**AI-powered blog generation platform** for Australian Police Check & Background Screening content.

Built with **LangGraph multi-agent pipeline** (9 nodes), **pgvector RAG**, **cross-encoder reranking**, and a **4-layer content quality engine** to produce publication-ready articles for [validex.com.au](https://validex.com.au).

---

## Table of Contents

- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Reference (35 Endpoints)](#api-reference)
- [Pipeline Architecture](#pipeline-architecture)
- [Database Schema](#database-schema)
- [Configuration Reference](#configuration-reference)
- [Deployment](#deployment)
- [Testing](#testing)
- [Maintenance Runbooks](#maintenance-runbooks)
- [License](#license)

---

## Architecture

```
                          ┌──────────────────────────────────┐
                          │    Angular 19 Frontend (SPA)     │
                          │   app.ts · chat.service.ts       │
                          │   auth.service.ts · markdown     │
                          └──────────────┬───────────────────┘
                                         │ HTTP / SSE
                          ┌──────────────▼───────────────────┐
                          │    FastAPI (api_server.py)        │
                          │  35 endpoints · JWT auth · CORS  │
                          │  Rate limiting · LLM semaphore   │
                          └──────────────┬───────────────────┘
                                         │
                ┌────────────────────────▼──────────────────────────┐
                │              main.py — Orchestrator                │
                │   Semantic Cache → LangGraph → sanitize_payload   │
                └────────────────────────┬──────────────────────────┘
                                         │
        ┌─────────────────────────────────▼──────────────────────────────────┐
        │                    LangGraph State Machine (9 Nodes)               │
        │                                                                    │
        │  Parser → Researcher → RAG Evaluator → Supervisor                 │
        │                                          ↓           ↓             │
        │                                      Writer    Deep Researcher     │
        │                                        ↓            ↓              │
        │                                      Editor ←──── Writer           │
        │                                        ↓                           │
        │                                   ML Quality Gate → ML Collector   │
        └────────────────────────────────────────────────────────────────────┘
                                         │
                ┌────────────────────────▼──────────────────────────┐
                │                  Data Layer                       │
                │  PostgreSQL + pgvector │ Redis │ Unsplash API     │
                └──────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Angular 19, TypeScript, SCSS |
| **Backend** | Python 3.11+, FastAPI, Uvicorn |
| **AI/ML Pipeline** | LangGraph, LangChain, Gemini / OpenAI |
| **Vector Database** | PostgreSQL + pgvector (HNSW index) |
| **Caching** | Redis (sessions, rate limits), Semantic Cache (embedding-based) |
| **Embeddings** | Google `text-embedding-004` / OpenAI `text-embedding-3-small` |
| **Local ML** | ONNX Cross-Encoder (reranker), Sentence-Transformers, NLI |
| **Auth** | JWT (HS256) + bcrypt |
| **Export** | python-docx, xhtml2pdf, markdown2 |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Required for `typing` features |
| Node.js | 18+ | For Angular frontend |
| PostgreSQL | 14+ | With `pgvector` extension enabled |
| Redis | 6+ | Optional but recommended for sessions |
| Google API Key | — | Free tier available via [AI Studio](https://aistudio.google.com/) |

---

## Quick Start

### 1. Clone & Setup Python Environment

```bash
git clone https://github.com/THVKhang/ChatBot_Validex.git
cd ChatBot_Validex

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your values (see Configuration Reference below)
```

**Minimum required variables:**
```env
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=postgresql://user:pass@host:5432/dbname?sslmode=require
JWT_SECRET_KEY=your_random_secret_string
USE_LIVE_LLM=1
```

### 3. Initialize Database

```bash
# Verify PostgreSQL connection
python -m app.check_pg_connection

# Create tables + indexes
psql "$DATABASE_URL" -f sql/database.sql

# Ingest knowledge base into pgvector
python -m app.ingest_pgvector

# Verify ingestion
python -m app.verify_pgvector_ingest
```

### 4. Start Application

**Option A: All-in-One** (recommended)
```bash
python run.py
# Starts backend on :8000 + frontend on :8001
```

**Option B: Separate Processes**
```bash
# Terminal 1: Backend
python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd ui/angular-frontend
npm install   # first time only
npm start
```

### 5. Open Browser

- **Frontend**: http://localhost:8001 (or http://localhost:4200 if running via `npm start` defaults)
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Health Check**: http://localhost:8000/api/health

### 6. Run Tests

```bash
# All tests
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_parser.py -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=html
```

---

## Project Structure

```
ChatBot_Validex/
├── app/                          # Backend application (51 modules)
│   ├── api_server.py             # FastAPI server (35 endpoints, 1605 lines)
│   ├── main.py                   # Central orchestrator (process_prompt)
│   ├── config.py                 # 50+ settings from .env
│   ├── auth.py                   # JWT + bcrypt authentication
│   │
│   ├── graph.py                  # LangGraph state machine definition
│   ├── graph_state.py            # Shared state schema (TypedDict)
│   │
│   ├── agents/                   # LangGraph node implementations
│   │   ├── parser_node.py        # Intent/topic extraction
│   │   ├── researcher_node.py    # Multi-query RAG retrieval
│   │   ├── rag_evaluator_node.py # Context quality scoring
│   │   ├── writer_node.py        # 3-stage blog generation + edit mode
│   │   ├── editor_node.py        # 4-layer quality gate
│   │   ├── ml_gate_node.py       # ML quality prediction
│   │   ├── ml_collector_node.py  # Training data collection
│   │   ├── discovery_agent.py    # AI-powered source discovery
│   │   └── scraper.py            # Web scraper for deep research
│   │
│   ├── llm/                      # LLM provider abstraction
│   │   ├── provider.py           # Multi-provider factory (Gemini/OpenAI/Groq)
│   │   └── token_tracker.py      # Per-user daily token budgets
│   │
│   ├── ml/                       # ML/DL training pipeline
│   │   ├── ml_data_collector.py  # Feature extraction
│   │   ├── ml_quality_gate.py    # Quality prediction model
│   │   ├── ml_trainer.py         # XGBoost / Random Forest trainer
│   │   ├── ml_feedback_loop.py   # Auto-retrain on new data
│   │   ├── embedding_trainer.py  # Fine-tune retrieval embeddings
│   │   ├── reranker_trainer.py   # Train cross-encoder
│   │   └── contrastive_data_builder.py  # Build training pairs
│   │
│   ├── langchain_pipeline.py     # Core LLM orchestration (142KB)
│   ├── generator.py              # Outline templates + blog assembly
│   ├── parser.py                 # Regex prompt parser
│   ├── reranker.py               # ONNX cross-encoder reranker
│   ├── rag_evaluator.py          # Multi-dimensional RAG scoring
│   ├── local_semantics.py        # Sentence-transformer embeddings
│   ├── local_nli.py              # NLI fact verification
│   │
│   ├── ingest_pgvector.py        # Vector ingestion to PostgreSQL
│   ├── collect_au_sources.py     # Australian gov data crawler (60KB)
│   ├── legal_chunker.py          # Jurisdiction-aware chunking
│   ├── metadata_enricher.py      # Auto-tag jurisdiction/topic
│   ├── worker.py                 # Cron jobs (ingestion, scheduling)
│   │
│   ├── session_manager.py        # In-memory session state
│   ├── session_store.py          # PostgreSQL session persistence
│   ├── redis_session.py          # Redis-backed sessions
│   ├── semantic_cache.py         # Embedding-based query dedup
│   ├── db_pool.py                # Connection pooling
│   ├── report_store.py           # CRUD for saved reports
│   ├── publisher.py              # Report → Markdown/HTML export
│   ├── prompt_guard.py           # Injection/jailbreak detection
│   └── analytics.py              # Token/quality/feedback analytics
│
├── ui/angular-frontend/          # Angular 19 SPA
│   └── src/app/
│       ├── app.ts                # Main component (44KB)
│       ├── app.html              # Template (42KB)
│       ├── app.scss              # Styles (84KB)
│       ├── chat.service.ts       # API client
│       ├── chat.models.ts        # TypeScript interfaces
│       ├── auth.service.ts       # Auth client
│       ├── auth.interceptor.ts   # JWT interceptor
│       └── markdown.pipe.ts      # Markdown → HTML renderer
│
├── tests/                        # 37 test files
├── sql/                          # Database schemas
│   ├── database.sql              # Main schema (pgvector + cache + tokens)
│   ├── sessions.sql              # Session tables
│   ├── mlops.sql                 # ML tracking tables
│   └── migrate_legal_metadata.sql
│
├── docs/                         # Weekly documentation
├── data/                         # Knowledge base data files
├── .env.example                  # Environment variable template
├── requirements.txt              # Python dependencies
├── run.py                        # All-in-one starter script
└── run_server.bat                # Windows shortcut
```

---

## API Reference

### Authentication (`/api/auth/*`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/auth/register` | Public | Register `{username, password}` |
| `POST` | `/api/auth/login` | Public | Login → `{access_token, token_type}` |

### Chat (`/api/chat/*`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/chat` | Optional | Generate blog (synchronous) |
| `POST` | `/api/chat/stream` | Optional | Generate blog (SSE streaming) |
| `POST` | `/api/chat/upload` | Required | Upload PDF/DOCX for context |
| `POST` | `/api/chat/export` | Public | Export markdown → DOCX/PDF/HTML |
| `GET` | `/api/chat/sessions` | Optional | List chat sessions |
| `GET` | `/api/chat/sessions/{id}` | Optional | Get session history |

### Reports (`/api/reports/*`)

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/reports` | Public | Save report |
| `GET` | `/api/reports` | Public | List reports |
| `GET` | `/api/reports/{id}` | Public | Get report detail |
| `DELETE` | `/api/reports/{id}` | Public | Delete report |
| `PATCH` | `/api/reports/{id}/status` | Public | Update status (`Draft→Reviewed→Approved`) |
| `POST` | `/api/reports/{id}/publish` | Public | Publish approved report |
| `POST` | `/api/reports/{id}/feedback` | Optional | Submit thumbs up/down |

### Admin (`/api/admin/*`) — Requires admin JWT

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/admin/ingest` | Trigger knowledge base re-ingestion |
| `GET` | `/api/admin/ingest/status` | Ingestion job status |
| `POST` | `/api/admin/discover` | AI Discovery Agent: find new sources |
| `GET` | `/api/admin/users` | List all users |
| `GET` | `/api/admin/crawl-history` | Crawl job logs |
| `GET` | `/api/admin/token-usage` | Token usage dashboard |
| `GET` | `/api/admin/pending-reviews` | HITL: pending article reviews |
| `POST` | `/api/admin/reviews/{run_id}` | HITL: approve/reject article |

### Analytics (`/api/admin/analytics/*`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/admin/analytics/tokens` | Token usage over time |
| `GET` | `/api/admin/analytics/quality` | Editor verdict distribution |
| `GET` | `/api/admin/analytics/cache` | Semantic cache statistics |
| `GET` | `/api/admin/analytics/feedback` | User feedback stats |

### Scheduling (`/api/admin/schedule/*`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/admin/schedule` | Create blog generation schedule |
| `GET` | `/api/admin/schedule` | List schedules |
| `DELETE` | `/api/admin/schedule/{id}` | Delete schedule |

### Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Deep health check (DB, Redis, Pool) |
| `GET` | `/api/metrics` | Application metrics (JSON) |
| `GET` | `/metrics` | Prometheus-compatible metrics |
| `GET` | `/api/user/token-budget` | User's remaining daily tokens |
| `GET` | `/api/source-analytics` | Knowledge base source stats |
| `GET` | `/api/knowledge/health` | Knowledge base health report |

---

## Pipeline Architecture

### LangGraph State Machine — 9 Nodes

```
START → Parser ──→ Researcher → RAG Evaluator → Supervisor
         │  (edit)                    ↕ retry        │
         ↓                                    ┌──────┴──────┐
       Writer ←─── Editor            Writer   Deep Researcher
         │           ↕ loop<3          ↑           │
         ↓                             └───────────┘
   ML Quality Gate → ML Collector → END
```

| Node | Module | Role |
|------|--------|------|
| **Parser** | `parser_node.py` | Extract intent, topic, tone, audience, length |
| **Researcher** | `researcher_node.py` | 8-step RAG: multi-query → pgvector → rerank → compress |
| **RAG Evaluator** | `rag_evaluator_node.py` | Score retrieval quality (relevance, coverage, diversity) |
| **Supervisor** | `graph.py` | Route by complexity: simple → Writer, complex → Deep Researcher |
| **Deep Researcher** | `graph.py` | Extended legal queries for complex topics |
| **Writer** | `writer_node.py` | 3-stage generation (Plan → Draft → Self-Review) + edit fast-path |
| **Editor** | `editor_node.py` | 4-layer quality gate (structural → SEO → fast-accept → LLM) |
| **ML Quality Gate** | `ml_gate_node.py` | ML model quality prediction (shadow/enforce modes) |
| **ML Collector** | `ml_collector_node.py` | Training data collection (passive, never blocks) |

### Circuit Breakers

| Breaker | Limit | Prevents |
|---------|-------|----------|
| Global Step | 8 total node visits | Infinite loops across nested retries |
| RAG Retry | 2 attempts | Researcher loops |
| Editor Loop | 3 iterations | Writer↔Editor cycles |
| ML Gate | 1 block | ML model re-routing |

---

## Database Schema

### Main Tables

| Table | File | Purpose |
|-------|------|---------|
| `validex_knowledge` | `sql/database.sql` | Knowledge chunks with 1536-dim embeddings |
| `validex_semantic_cache` | `sql/database.sql` | Semantic cache (384-dim, local embeddings) |
| `token_usage_log` | `sql/database.sql` | Token consumption tracking |
| `users` | `sql/sessions.sql` | User accounts (username, bcrypt hash, admin flag) |
| `chat_sessions` | `sql/sessions.sql` | Session history (JSONB turns) |
| `reports` | auto-created | Saved blog reports |
| `blog_schedule` | auto-created | Cron-based generation schedules |
| `crawl_logs` | auto-created | Data ingestion job logs |
| `prompt_ab_tests` | `sql/mlops.sql` | AB test / HITL review tracking |

### Key Indexes

```sql
-- HNSW vector index for fast similarity search
CREATE INDEX idx_validex_knowledge_embedding_hnsw
ON validex_knowledge USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Full-text search index
CREATE INDEX idx_validex_knowledge_fts
ON validex_knowledge USING GIN (fts_content);

-- Composite filter index (retriever's most common pattern)
CREATE INDEX idx_validex_knowledge_status_jurisdiction
ON validex_knowledge(status, jurisdiction);
```

---

## Configuration Reference

All settings in [app/config.py](app/config.py), loaded from `.env`. See [.env.example](.env.example) for full template.

### Critical Settings

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `GOOGLE_API_KEY` | — | **Yes** | Gemini API key |
| `DATABASE_URL` | — | **Yes** | PostgreSQL connection string |
| `JWT_SECRET_KEY` | — | **Yes** | JWT signing secret |
| `USE_LIVE_LLM` | `0` | — | Enable live LLM calls (`1` for production) |

### LLM Models

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `auto` | `auto` / `google` / `openai` |
| `LLM_MODEL_PRO` | `gemini-2.5-flash` | Writer model (creative, temp 0.7) |
| `LLM_MODEL_FAST` | `gemini-2.0-flash-lite` | Parser/planner model (fast, temp 0.3) |
| `EDITOR_TEMPERATURE` | `0.1` | Editor evaluation temperature |
| `WRITER_TEMPERATURE` | `0.7` | Writer generation temperature |

### Rate Limiting

| Variable | Default | Description |
|----------|---------|-------------|
| `RATE_LIMIT_PER_MINUTE` | `30` | Max requests per IP per minute |
| `BURST_LIMIT_PER_10S` | `5` | Max burst per 10 seconds |
| `MAX_CONCURRENT_PER_IP` | `3` | Max concurrent per IP |
| `MAX_CONCURRENT_LLM_REQUESTS` | `10` | Global LLM semaphore |
| `REQUEST_TIMEOUT_SECONDS` | `180` | Per-request timeout |

### Token Budgets

| Variable | Default | Description |
|----------|---------|-------------|
| `QUOTA_FREE` | `25,000` | Daily tokens for free tier |
| `QUOTA_STARTER` | `80,000` | Daily tokens for starter tier |
| `QUOTA_PRO` | `250,000` | Daily tokens for pro tier |
| `QUOTA_ENTERPRISE` | `1,000,000` | Daily tokens for enterprise tier |

---

## Deployment

### Production Checklist

- [ ] Set `USE_LIVE_LLM=1`
- [ ] Set `USE_AGENTIC_RAG=1`
- [ ] Set strong `JWT_SECRET_KEY`
- [ ] Set `ALLOWED_ORIGINS` to production domain
- [ ] Run `sql/database.sql` on production database
- [ ] Run `python -m app.ingest_pgvector` to populate knowledge base
- [ ] Configure Redis for sessions (`USE_REDIS_SESSIONS=1`)
- [ ] Set up reverse proxy (nginx) for HTTPS
- [ ] Build Angular frontend: `cd ui/angular-frontend && npm run build`
- [ ] Set `LANGCHAIN_TRACING_V2=true` for LangSmith monitoring

### Docker (Optional)

```bash
# Backend
uvicorn app.api_server:app --host 0.0.0.0 --port 8000

# Frontend (serve built assets)
# Built files go to ui/angular-frontend/dist/angular-frontend/browser/
# api_server.py auto-serves these at the catch-all route
```

### CI/CD

| Workflow | Schedule | Description |
|----------|----------|-------------|
| `delta_sync.yml` | Weekly (Sun 2AM UTC) | Crawl new sources → chunk → embed → pgvector |

---

## Testing

37 test files organized by category:

| Category | Tests | Coverage |
|----------|-------|----------|
| Pipeline | `test_e2e_pipeline`, `test_main_flow` | End-to-end generation |
| Graph | `test_graph_state_machine`, `test_circuit_breaker` | Routing + breakers |
| Parser | `test_parser`, `test_prompt_parser_llm`, `test_prompt_edit_constraints` | Intent detection |
| Retrieval | `test_retriever`, `test_hybrid_fallback`, `test_semantic_cache` | RAG + fallback |
| Generation | `test_generator`, `test_structured_output`, `test_token_budgeting` | Templates + budgets |
| Quality | `test_editor`, `test_output_fixes`, `test_accuracy_evaluation` | Quality gate |
| Data | `test_hierarchical_chunking`, `test_golden_facts`, `test_knowledge_gap` | Ingestion + knowledge |
| API | `test_api_edge_cases`, `test_api_reports`, `test_export` | Endpoints |
| Security | `test_auth_security`, `test_prompt_guard_adversarial` | Auth + injection |
| Scalability | `test_concurrency`, `test_scalability_simulation` | Load testing |

```bash
# Run all tests
python -m pytest tests/ -q

# Run specific category
python -m pytest tests/ -k "test_parser or test_graph" -v

# Run with coverage report
python -m pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

---

## Maintenance Runbooks

### Re-ingest Knowledge Base

```bash
# 1. Crawl new Australian sources
python -m app.collect_au_sources

# 2. Ingest into pgvector
python -m app.ingest_pgvector

# 3. Verify
python -m app.verify_pgvector_ingest
```

### Clear Semantic Cache

```bash
psql "$DATABASE_URL" -c "TRUNCATE validex_semantic_cache;"
```

### Re-embed Stale Chunks

```bash
python -m app.refresh_embeddings
```

### Train ML Quality Model

```bash
python -m app.ml.ml_trainer
# Auto-promote shadow → enforce when F1 > 0.85
```

### Run Gap Analysis

```bash
python run_gap_analysis.py
# Identifies topics missing from knowledge base
```

### Database Maintenance

```bash
# Check connection pool status
curl http://localhost:8000/api/health | jq '.connection_pool'

# Vacuum pgvector table
psql "$DATABASE_URL" -c "VACUUM ANALYZE validex_knowledge;"

# Reindex HNSW (after large ingestion)
psql "$DATABASE_URL" -c "REINDEX INDEX idx_validex_knowledge_embedding_hnsw;"
```

### Promote User to Admin

```bash
psql "$DATABASE_URL" -c "UPDATE users SET is_admin = TRUE WHERE username = 'your_username';"
```

---

## Documentation

- **API Reference**: [docs/API.md](docs/API.md) — All 35 endpoints with request/response examples
- **Deployment Guide**: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) — Setup, runbooks, troubleshooting
- **Environment Template**: [.env.example](.env.example) — All configuration variables
- **Database Schema**: [sql/database.sql](sql/database.sql) — Tables, indexes, pgvector setup
- **API Swagger**: http://localhost:8000/docs (when server is running)

---

## License

[MIT](LICENSE)
