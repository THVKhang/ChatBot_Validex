# Deployment & Handover Guide

> Step-by-step guide for deploying and maintaining the Validex AI Blog Generator.
> Last updated: 2026-07-27

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Infrastructure Setup](#2-infrastructure-setup)
3. [Application Deployment](#3-application-deployment)
4. [Post-Deployment Verification](#4-post-deployment-verification)
5. [Operational Runbooks](#5-operational-runbooks)
6. [Monitoring & Alerts](#6-monitoring--alerts)
7. [Troubleshooting Guide](#7-troubleshooting-guide)
8. [Key Contacts & Handover Notes](#8-key-contacts--handover-notes)

---

## 1. System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB (local ML models loaded in memory) |
| Disk | 5 GB | 20 GB (knowledge base + logs) |

### Software

| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Node.js | 18+ | Angular frontend build |
| PostgreSQL | 14+ | Vector database + sessions |
| pgvector extension | 0.5+ | Embedding similarity search |
| Redis | 6+ | Sessions, rate limiting (optional) |

### External Services

| Service | Required? | Purpose |
|---------|-----------|---------|
| Google AI Studio (Gemini) | **Yes** | LLM generation + embeddings |
| Unsplash API | Optional | Blog header images |
| Google Custom Search | Optional | AI Discovery Agent |
| LangSmith | Optional | LLM tracing/debugging |

---

## 2. Infrastructure Setup

### 2.1 PostgreSQL + pgvector

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Run the schema
\i sql/database.sql
\i sql/sessions.sql
\i sql/mlops.sql
```

Verify:
```bash
python -m app.check_pg_connection
# Expected: "✅ Connection OK. pgvector extension: installed"
```

### 2.2 Redis (Optional)

```bash
# Start Redis
redis-server

# Verify
redis-cli ping
# Expected: PONG
```

Enable in `.env`:
```env
REDIS_URL=redis://localhost:6379/0
USE_REDIS_SESSIONS=1
USE_REDIS_RATE_LIMIT=1
```

### 2.3 Knowledge Base Ingestion

```bash
# Step 1: Crawl Australian government sources
python -m app.collect_au_sources

# Step 2: Ingest into pgvector
python -m app.ingest_pgvector

# Step 3: Verify ingestion
python -m app.verify_pgvector_ingest
# Expected: "✅ N chunks ingested, M with valid embeddings"
```

---

## 3. Application Deployment

### 3.1 Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with production values

# Start server
uvicorn app.api_server:app --host 0.0.0.0 --port 8000 --workers 4

# Or with auto-reload (development)
uvicorn app.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 3.2 Frontend

```bash
cd ui/angular-frontend

# Install dependencies
npm install

# Development
npm start

# Production build
npm run build
# Output: dist/angular-frontend/browser/
# api_server.py auto-serves these files at catch-all route
```

### 3.3 Production `.env` Checklist

```env
# MUST CHANGE
GOOGLE_API_KEY=<real key>
DATABASE_URL=<real connection string>
JWT_SECRET_KEY=<random 64-char hex>
USE_LIVE_LLM=1

# SHOULD CHANGE
ALLOWED_ORIGINS=https://yourdomain.com
USE_AGENTIC_RAG=1
USE_REDIS_SESSIONS=1
LANGCHAIN_TRACING_V2=true

# REVIEW
RATE_LIMIT_PER_MINUTE=30
MAX_CONCURRENT_LLM_REQUESTS=10
REQUEST_TIMEOUT_SECONDS=180
```

### 3.4 Nginx Reverse Proxy

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate     /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;

    # API and SSE
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;  # Allow long LLM requests
    }

    # Prometheus metrics
    location /metrics {
        proxy_pass http://127.0.0.1:8000;
    }

    # Frontend SPA
    location / {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

---

## 4. Post-Deployment Verification

### Health Check
```bash
curl http://localhost:8000/api/health | python -m json.tool
```
Expected: `"status": "ok"`, all subsystems green.

### Generate a Test Blog
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write about police checks in Australia"}'
```
Expected: Full JSON response with `generated.draft` containing markdown blog.

### Register & Login
```bash
# Register
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Login
curl -X POST http://localhost:8000/api/auth/login \
  -d "username=admin&password=admin123"

# Promote to admin
psql "$DATABASE_URL" -c "UPDATE users SET is_admin = TRUE WHERE username = 'admin';"
```

### Run Test Suite
```bash
python -m pytest tests/ -q
# Expected: 54+ tests passed
```

---

## 5. Operational Runbooks

### 5.1 Weekly Knowledge Base Update

Runs automatically via CI/CD (`delta_sync.yml`) every Sunday 2 AM UTC.

Manual trigger:
```bash
# Via API (admin required)
curl -X POST http://localhost:8000/api/admin/ingest \
  -H "Authorization: Bearer <admin_token>"

# Check status
curl http://localhost:8000/api/admin/ingest/status \
  -H "Authorization: Bearer <admin_token>"
```

### 5.2 AI Discovery Agent

Find new Australian government data sources:
```bash
# Requires GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX
curl -X POST http://localhost:8000/api/admin/discover \
  -H "Authorization: Bearer <admin_token>"
```

### 5.3 Clear Semantic Cache

When knowledge base is updated, stale cached responses may be served:
```bash
psql "$DATABASE_URL" -c "TRUNCATE validex_semantic_cache;"
```

### 5.4 Re-embed Stale Chunks

After changing embedding model or detecting quality issues:
```bash
python -m app.refresh_embeddings
```

### 5.5 Train ML Quality Model

```bash
# Collect enough data first (100+ pipeline runs recommended)
python -m app.ml.ml_trainer

# Check model status
# ML_GATE_MODE=shadow (collecting data, never blocks)
# ML_GATE_MODE=enforce (auto-rejects when confidence > 80%)
```

### 5.6 Database Maintenance

```bash
# Vacuum & analyze (run weekly)
psql "$DATABASE_URL" -c "VACUUM ANALYZE validex_knowledge;"

# Reindex HNSW after large ingestion
psql "$DATABASE_URL" -c "REINDEX INDEX idx_validex_knowledge_embedding_hnsw;"

# Check table sizes
psql "$DATABASE_URL" -c "SELECT pg_size_pretty(pg_total_relation_size('validex_knowledge'));"

# Connection pool status
curl http://localhost:8000/api/health | python -m json.tool | grep -A5 connection_pool
```

### 5.7 HITL (Human-in-the-Loop) Review

Articles flagged by the quality gate appear in the admin panel:
```
Frontend → Settings → Admin Panel → Pending Reviews
```
Or via API:
```bash
curl http://localhost:8000/api/admin/pending-reviews \
  -H "Authorization: Bearer <admin_token>"
```

---

## 6. Monitoring & Alerts

### Key Metrics

| Metric | Endpoint | Alert Threshold |
|--------|----------|----------------|
| System health | `GET /api/health` | `status != "ok"` |
| Error rate | `GET /api/metrics` | `chat_errors_total / chat_requests_total > 10%` |
| Latency | `GET /api/metrics` | P95 > 30s |
| LLM availability | `GET /api/health` | `runtime.llm_available = false` |
| DB connectivity | `GET /api/health` | `database.status = "error"` |
| Knowledge freshness | `GET /api/knowledge/health` | `stale_chunks_percentage > 20%` |

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'validex'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Log Files

| Log | Location | Content |
|-----|----------|---------|
| Application | stdout (uvicorn) | Structured JSON logs |
| Error log | `_error.log` | Unhandled exceptions |
| LangSmith | cloud dashboard | Full LLM trace with prompts/responses |

---

## 7. Troubleshooting Guide

### "Cannot reach the API"
- Check backend is running: `curl http://localhost:8000/api/health`
- Check CORS: `ALLOWED_ORIGINS` must include frontend URL
- Check firewall/port forwarding

### "LLM not available"
- Verify `GOOGLE_API_KEY` is set and valid
- Check `USE_LIVE_LLM=1`
- Test API key: `python -c "import google.generativeai as genai; genai.configure(api_key='your_key'); print(genai.list_models())"`

### "No retrieval results"
- Check pgvector: `python -m app.verify_pgvector_ingest`
- Check `USE_PGVECTOR_RETRIEVAL=1`
- Verify embeddings match model: `EMBEDDING_PROVIDER` must match ingested embeddings
- If mismatch: truncate table and re-ingest

### "Session not persisting"
- Check `DATABASE_URL` is configured
- Verify `chat_sessions` table exists: `psql "$DATABASE_URL" -c "SELECT count(*) FROM chat_sessions;"`

### "Rate limit errors (429)"
- Increase `RATE_LIMIT_PER_MINUTE` in `.env`
- Or disable: `USE_RATE_LIMIT=0` (not recommended for production)

### "Edit mode generates new blog instead of editing"
- Fixed in Bug Audit (2026-07-26): fallthrough trap eliminated
- Verify `writer_node.py` has the safe-return pattern at line 484

### "UI artifacts in output (edit_note, thumb_up)"
- Fixed in Bug Audit (2026-07-26): 7-pass sanitizer
- Verify `main.py` `_sanitize_ui_artifacts()` has 7 passes

---

## 8. Key Contacts & Handover Notes

### Repository
- **GitHub**: `THVKhang/ChatBot_Validex`
- **Branch**: `fix/audit-and-agent-upgrade` (latest fixes)

### Key Documentation
| Document | Location | Content |
|----------|----------|---------|
| README | `README.md` | Project overview, setup, structure |
| API Reference | `docs/API.md` | All 35 endpoints with examples |
| Pipeline Walkthrough | Walkthrough artifact | Full architectural deep-dive |
| Environment Template | `.env.example` | All configuration variables |
| Database Schema | `sql/database.sql` | Tables, indexes, pgvector setup |
| Deployment Guide | `docs/DEPLOYMENT.md` | This document |

### Architecture Decisions

1. **LangGraph over LangChain Agents**: Deterministic routing with explicit state machine vs. autonomous agent loops. More predictable, debuggable.
2. **ONNX Reranker over API**: Zero-token, ~50ms latency, no API dependency.
3. **Debate Agent Pattern**: Writer (temp=0.7) and Editor (temp=0.1) use different LLM configurations to prevent self-confirmation bias.
4. **Shadow→Enforce ML Gate**: Collect training data before auto-rejecting. Prevents premature blocking.
5. **Edit fast-path**: Follow-up prompts skip RAG entirely, preserving context and speed.
6. **7-pass sanitizer**: Defense-in-depth against web scraping artifacts leaking into output.

### Known Limitations

1. **Embedding model lock-in**: Changing embedding model requires full re-ingestion (`TRUNCATE validex_knowledge; python -m app.ingest_pgvector`)
2. **Session memory depth**: Limited to 5 turns (`MAX_CONVERSATION_TURNS`). Older turns are dropped from LLM context.
3. **Single-language RAG**: Knowledge base is English-only (Australian sources). Vietnamese prompts work but retrieval quality degrades.
4. **ML Gate**: Currently in `shadow` mode (collecting data). Needs 100+ pipeline runs before `enforce` mode is reliable.
