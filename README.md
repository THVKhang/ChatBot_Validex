# ChatBot Validex

AI-powered blog generation platform for Australian Police Check and Background Screening content. Built with a **multi-agent RAG pipeline** (LangGraph), **cross-encoder reranking**, and a **multi-layer content quality engine** to produce publication-ready articles for [validex.com.au](https://validex.com.au).

## Architecture Overview

```
User Prompt
    │
    ▼
┌─────────┐    ┌────────────┐    ┌───────────────┐    ┌────────────┐
│  Parser  │───▶│ Researcher │───▶│ RAG Evaluator │───▶│ Supervisor │
│  (NLP)   │    │ (Multi-Q)  │    │  (Grounding)  │    │  (Router)  │
└─────────┘    └────────────┘    └───────────────┘    └────────────┘
                     │                                       │
              ┌──────┴──────┐                         ┌──────┴──────┐
              │  Reranker   │                         │   Writer    │
              │(CrossEnc.)  │                         │ (3-Stage)   │
              └─────────────┘                         └──────┬──────┘
                                                             │
                                                      ┌──────┴──────┐
                                                      │   Editor    │
                                                      │ (Hybrid QA) │
                                                      └─────────────┘
                                                             │
                                                             ▼
                                                      Published Blog
```

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run backend API

**Option 1: The simple way (Starts BOTH Backend and Frontend)**
Just run the master script. It will auto-detect available ports and start everything:
```bash
python run.py
```

**Option 2: Using uvicorn directly**
```bash
# Windows
.\.venv\Scripts\activate
python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000

# macOS / Linux
source .venv/bin/activate
python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000
```

3. Run frontend

```bash
cd ui/angular-frontend
npm install  # (Only needed the first time)
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

## Agentic RAG Pipeline (LangGraph)

The system uses a **LangGraph StateGraph** with specialized agent nodes. Each node is a focused expert that transforms the shared state:

| Node | Role | Key Features |
|------|------|-------------|
| **Parser** | NLP intent extraction | Detects topic, tone, audience, length, how-to vs informational intent |
| **Researcher** | Multi-query retrieval + web search | Query expansion via local semantics, deep scraping, research cache |
| **Reranker** | Cross-encoder relevance scoring | ms-marco-MiniLM-L-12-v2 (ONNX, CPU, ~50ms), topic isolation penalty |
| **RAG Evaluator** | Grounding quality gate | Validates retrieved context is sufficient before generation |
| **Supervisor** | Agentic loop controller | Routes to Writer or back to Researcher (max 4 iterations) |
| **Writer** | 3-stage blog generation | Stage 1: Outline planning → Stage 2: Chunked parallel generation → Stage 3: Self-review |
| **Editor** | Hybrid quality assurance | Layer 0: Code-based structural checks → Layer 1: SEO analysis → Layer 2: LLM evaluation |

### Writer 3-Stage Pipeline

```
Stage 1: _plan_outline()
  ├── Detect how-to intent → force Step 1, Step 2, Step 3...
  ├── Topic focus: "Stay on {topic}, don't mix check types"
  └── Anti-repetition: "Each section covers DIFFERENT aspects"

Stage 2: _generate() — Chunked parallel generation
  ├── Per-section LLM calls with assigned key points
  └── Mandatory outline injection: "Cover ONLY assigned points"

Stage 3: _self_review()
  ├── 9 quality checks: accuracy, citations, coherence, completeness,
  │   tone, format compliance, no repetition, topic focus, no file paths
  └── Rewrites draft if structural issues detected
```

### Editor Hybrid Quality Gate

```
Layer 0: Code-based structural checks (0 LLM tokens)
  ├── E01: Repetition detection (4-gram analysis)
  ├── E03: Length compliance (min/max word count)
  ├── E04: Heading structure (minimum ## sections)
  ├── E05: Paragraph balance (thin paragraph detection)
  ├── E06: Conclusion check
  ├── E08: File path leak detection (file://C:/ patterns)
  ├── E09: Step-by-step compliance (verify numbered steps when requested)
  └── E10: Repetitive intro detection (>60% word overlap between openers)

Layer 1: SEO + Readability checks (0 LLM tokens)
  └── Keyword density, heading hierarchy, meta tag analysis

Layer 2: LLM evaluation (uses separate Editor model)
  └── Debate Agent pattern: different model/temperature from Writer
```

### Debate Agent Pattern

The Writer and Editor use **different LLM configurations** to prevent self-confirmation bias:

| Agent | Model | Temperature | Purpose |
|-------|-------|-------------|---------|
| Writer | `GOOGLE_MODEL_NAME` (gemini-2.5-flash) | 0.7 (creative) | Prose quality, varied vocabulary |
| Editor | `EDITOR_GOOGLE_MODEL_NAME` (configurable) | 0.1 (strict) | Objective evaluation, consistent scoring |

## Content Quality Engine

The system enforces content quality through multiple defense layers:

### Source Path Sanitization (3-layer)
1. **Reference builder** (`_doc_reference_line`): Converts `file://C:/...` paths to `validex.com.au` or `act_name` references
2. **Draft post-processor** (`main.py`): Regex strips any remaining `file://` URLs and `C:\` paths
3. **Sources list sanitizer** (`main.py`): Extracts filenames from local paths in `sources_used`

### Topic Isolation
1. **Retriever penalty**: WWCC docs get 0.4× relevance score when query is about police checks (and vice versa)
2. **System prompt rule**: LLM instructed to focus ONLY on the check type in the user's question
3. **Self-review check**: Removes off-topic content about unrelated check types

### Anti-Repetition
1. **Outline injection**: "Cover ONLY assigned key points. Do NOT repeat across sections"
2. **Self-review check #7**: Detects and removes duplicate facts
3. **Editor E10**: Flags sections with >60% word overlap in opening sentences
4. **System prompt guardrail**: "Every paragraph must be analytically distinct"

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

| Component | Files | Description |
|-----------|-------|-------------|
| **Backend API** | [app/api_server.py](app/api_server.py) | FastAPI server, rate limiting, auth, CORS |
| **RAG Pipeline** | [app/langchain_pipeline.py](app/langchain_pipeline.py) | Core LLM orchestration, prompt templates, structured output |
| **Agent Graph** | [app/graph.py](app/graph.py), [app/graph_state.py](app/graph_state.py) | LangGraph StateGraph definition |
| **Writer Agent** | [app/agents/writer_node.py](app/agents/writer_node.py) | 3-stage blog generation (plan → generate → self-review) |
| **Editor Agent** | [app/agents/editor_node.py](app/agents/editor_node.py) | Hybrid quality gate (code + LLM) |
| **Researcher Agent** | [app/agents/researcher_node.py](app/agents/researcher_node.py) | Multi-query retrieval + web search + scraping |
| **Reranker** | [app/reranker.py](app/reranker.py) | Cross-encoder reranking (FlashRank ONNX) |
| **Local Semantics** | [app/local_semantics.py](app/local_semantics.py) | Sentence-transformer embeddings for 0-token operations |
| **Data Collection** | [app/collect_au_sources.py](app/collect_au_sources.py) | Australian government source scraper |
| **Vector Ingestion** | [app/ingest_pgvector.py](app/ingest_pgvector.py) | PostgreSQL pgvector chunk ingestion |
| **Frontend** | [ui/angular-frontend/](ui/angular-frontend/) | Angular dashboard with blog preview |

## Environment Variables

Feature behavior is controlled by environment variables in [app/config.py](app/config.py).

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `auto` | `google`, `openai`, or `auto` |
| `GOOGLE_MODEL_NAME` | `models/gemini-2.5-flash` | Primary generation model |
| `GOOGLE_FAST_MODEL_NAME` | `models/gemini-2.0-flash-lite` | Fast model for planning/review |
| `USE_LIVE_LLM` | `0` | Enable live LLM calls (set to `1` for production) |
| `USE_PGVECTOR_RETRIEVAL` | `1` | Use PostgreSQL pgvector for retrieval |
| `USE_STRUCTURED_OUTPUT` | `0` | Enable Pydantic structured output (fallback path) |
| `USE_AGENTIC_RAG` | `1` | Enable multi-agent LangGraph pipeline |
| `USE_UNSPLASH_IMAGES` | `1` | Enable Unsplash image lookup per section |
| `ALLOW_HYBRID_FALLBACK` | `1` | Allow web search when RAG context insufficient |
| `ENFORCE_QUALITY_GATE` | `1` | Enable Editor quality checks |
| `WRITER_TEMPERATURE` | `0.7` | Creative temperature for Writer |
| `EDITOR_TEMPERATURE` | `0.1` | Strict temperature for Editor |

### Hybrid Fallback Behavior

When retrieval is `out_of_domain` or `low_confidence`:
1. `ALLOW_HYBRID_FALLBACK=1` enables generation using web search + general knowledge.
2. Generated draft includes `HYBRID_WARNING_TEXT` as a transparency contract.
3. Runtime metadata includes `runtime.external_knowledge_used=true`.
4. Set `ALLOW_HYBRID_FALLBACK=0` to restore strict "Need More Context" behavior.

## Structured Output and Auto Images

The generation pipeline supports structured JSON output and section-level image lookup.

1. Structured Output (`with_structured_output` + Pydantic):
   1. LLM is asked to return schema fields: `title`, `introduction`, `sections[]`, `conclusion`, `meta_tags`.
   2. Each section requires `header`, `content`, `image_search_keyword`.
   3. Backend renders this structure into markdown with deterministic headings and blocks.

2. Unsplash Tool / Function Calling flow:
   1. LLM (or agent) proposes `image_search_keyword` per section.
   2. Backend calls Unsplash Search API to fetch a real image URL.
   3. If Unsplash is unavailable or API key is missing, system falls back to keyword-based `source.unsplash.com` image URL.

3. Environment variables:
   1. `USE_UNSPLASH_IMAGES=1`
   2. `UNSPLASH_ACCESS_KEY=...`
   3. `UNSPLASH_API_BASE=https://api.unsplash.com`
   4. `UNSPLASH_TIMEOUT_SECONDS=8`

## Token Budgeting for RAG Blog Length

The pipeline estimates token budget per prompt length and adjusts retrieval depth automatically.

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

## CI/CD

| Workflow | Schedule | Description |
|----------|----------|-------------|
| [delta_sync.yml](.github/workflows/delta_sync.yml) | Weekly (Sun 2AM UTC) | Syncs new Australian government sources, re-embeds, updates pgvector |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Generate blog from prompt |
| `/api/health` | GET | Server health + runtime mode |
| `/api/metrics` | GET | Generation metrics (latency, cache hit rate) |
| `/api/sessions` | GET | Active session listing |
| `/api/admin/cache/clear` | POST | Clear semantic cache (requires admin key) |

## License

[MIT](LICENSE)
