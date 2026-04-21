# ChatBot Validex

AI-assisted blog generation system for Police Check and Background Check content, with retrieval, generation, and runtime observability.

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run backend API

You must run the server using your virtual environment to ensure all dependencies like `psycopg` are found.

**Option A (Directly using venv):**
```bash
# Windows
.\.venv\Scripts\python.exe -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000

# macOS / Linux
./.venv/bin/python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000
```

**Option B (Activate venv first):**
```bash
# Windows
.\.venv\Scripts\activate
python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000
```

3. Run frontend

```bash
cd ui/angular-frontend
npm install
npm start
```

4. Run tests

```bash
# Windows
.\.venv\Scripts\python.exe -m pytest -q

# macOS / Linux
./.venv/bin/python -m pytest -q
```

## Gemini API Setup (Google AI Studio)

Use this mode when you want free-tier testing for generation + embeddings with Gemini.

1. Install integration package:

```bash
pip install langchain-google-genai
```

2. Update `.env`:
1. `GOOGLE_API_KEY=your_google_key`
2. `LLM_PROVIDER=google`
3. `EMBEDDING_PROVIDER=google`
4. `GOOGLE_MODEL_NAME=models/gemini-2.5-flash`
5. `GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001`
6. `USE_LIVE_LLM=1`

3. Notes:
1. `OPENAI_API_KEY` can stay empty when using Google provider.
2. Structured output still works with `USE_STRUCTURED_OUTPUT=1`.
3. If pgvector dimensions differ from previous embeddings, recreate table or re-ingest consistently.

## Documentation

Main weekly documents:
1. [Week 1](docs/week1.md)
2. [Week 2](docs/week2.md)
3. [Week 3](docs/week3.md)

Detailed weekly breakdown:
1. [Week 1 folder](docs/week1)
2. [Week 2 folder](docs/week2)
3. [Week 3 folder](docs/week3)

## Core Components

1. Backend API and orchestration: [app/api_server.py](app/api_server.py), [app/langchain_pipeline.py](app/langchain_pipeline.py)
2. Data collection and ingestion: [app/collect_au_sources.py](app/collect_au_sources.py), [app/ingest_pgvector.py](app/ingest_pgvector.py)
3. Frontend dashboard: [ui/angular-frontend/src/app](ui/angular-frontend/src/app)

## Notes

1. Runtime mode and metrics are available at /api/health and /api/metrics.
2. Feature behavior is controlled by environment variables in [app/config.py](app/config.py).
3. When retrieval is `out_of_domain` or `low_confidence`, the default behavior is Hybrid fallback:
1. `ALLOW_HYBRID_FALLBACK=1` enables generation using general knowledge when RAG grounding is insufficient.
2. Generated draft must start with `HYBRID_WARNING_TEXT` as a transparency contract.
3. Runtime metadata includes `runtime.external_knowledge_used=true` when this fallback is used.
4. Set `ALLOW_HYBRID_FALLBACK=0` to restore strict "Need More Context" behavior.

## Structured Output and Auto Images

The generation pipeline now supports structured JSON output and section-level image lookup.

1. Structured Output (`with_structured_output` + Pydantic):
1. LLM is asked to return schema fields: `title`, `introduction`, `sections[]`, `conclusion`, `meta_tags`.
2. Each section requires `header`, `content`, `image_search_keyword`.
3. Backend renders this structure into markdown with deterministic headings and blocks.

2. Unsplash Tool / Function Calling flow:
1. LLM (or agent) proposes `image_search_keyword` per section.
2. Backend calls Unsplash Search API to fetch a real image URL.
3. If Unsplash is unavailable or API key is missing, system falls back to keyword-based `source.unsplash.com` image URL.

3. Environment variables:
1. `USE_STRUCTURED_OUTPUT=1`
2. `USE_UNSPLASH_IMAGES=1`
3. `UNSPLASH_ACCESS_KEY=...`
4. `UNSPLASH_API_BASE=https://api.unsplash.com`
5. `UNSPLASH_TIMEOUT_SECONDS=8`

## Token Budgeting for RAG Blog Length

The pipeline now estimates token budget per prompt length and adjusts retrieval depth automatically.

1. Output token targets:
1. Short (`400 chu`) -> about `600` output tokens
2. Medium (`800 chu`) -> about `1200` output tokens
3. Long (`1200 chu`) -> about `1800` output tokens

2. Input token budget rule:
1. `INPUT_OUTPUT_RATIO_MIN=1.5`
2. `INPUT_OUTPUT_RATIO_MAX=2.0`
3. This means the system targets input context around `1.5x` to `2.0x` of output size.

3. Dynamic retrieval `TOP_K` by length profile:
1. Short: `TOP_K_SHORT_MIN=3`, `TOP_K_SHORT_MAX=4`
2. Medium: `TOP_K_MEDIUM_MIN=6`, `TOP_K_MEDIUM_MAX=8`
3. Long: `TOP_K_LONG_MIN=10`, `TOP_K_LONG_MAX=12`

4. Runtime diagnostics in API response:
1. `runtime.token_budget.output_tokens_target`
2. `runtime.token_budget.output_tokens_estimated`
3. `runtime.token_budget.input_tokens_target_min`, `input_tokens_target`, `input_tokens_target_max`
4. `runtime.token_budget.input_tokens_estimated`
5. `runtime.token_budget.recommended_top_k`
6. `runtime.token_budget.input_budget_sufficient`

## Vector Ingestion Checks

1. Preflight database connectivity and pgvector extension:

```bash
python -m app.check_pg_connection
```

2. Ingest canonical JSONL chunks into PostgreSQL vector table:

```bash
python -m app.ingest_pgvector
```

3. Verify ingestion result (rows, indexes, vector sanity):

```bash
python -m app.verify_pgvector_ingest
```

4. Apply SQL bootstrap (extension + table + indexes) if needed:

```bash
psql "$DATABASE_URL" -f sql/database.sql
```

5. Runtime retrieval order:
1. PostgreSQL pgvector (when `USE_PGVECTOR_RETRIEVAL=1` and DB is reachable)
2. Pinecone (when enabled)
3. Local guarded retrieval fallback
