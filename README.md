# ChatBot Validex

AI-assisted blog generation system for Police Check and Background Check content, with retrieval, generation, and runtime observability.

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Run backend API

```bash
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
python -m pytest -q
```

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
