# Data Workflow for Agentic RAG

## Folder Layout
- data/raw: Original source files (.txt, .pdf)
- data/processed: Cleaned text files for retrieval
- data/metadata: Metadata map (documents.json)
- data/benchmark: Benchmark queries and reports
- data/goldens: Golden reference answers for generation quality

## Step 1: Collect and Name Raw Files
Store source files in data/raw using this naming rule:
- [type]_[topic]_[source].txt

Example:
- policy_spent_conviction_nsw_gov.txt
- faq_processing_time_validex.txt

## Step 2: Ingest and Vectorize
For local retrieval pipeline:
1. Run ingest from raw to processed + metadata:
   - python -m app.ingest_data

For pgvector pipeline:
1. Collect canonical chunks (HTML/PDF/text) to JSONL:
   - python -m app.collect_au_sources
2. Embed and upsert to PostgreSQL with metadata:
   - python -m app.ingest_pgvector

## Step 3: Benchmark Retrieval
Run retrieval benchmark on data/benchmark/retrieval_queries.json:
- python -m app.evaluate_benchmark

Output report:
- data/benchmark/retrieval_report.md

## Step 4: Evaluate Against Goldens
Run generation quality check against goldens:
- python -m app.evaluate_goldens

Input files:
- data/benchmark/generation_queries.json
- data/goldens/golden_answers.json

Output report:
- data/benchmark/golden_report.md

## Metadata Standard (documents.json)
Each record should include:
- id
- file_stem
- title
- topic
- jurisdiction
- authority_score
- approved
- source_url
- last_updated

Why authority_score matters:
- When two sources conflict, the system can prioritize the more reliable source (for example ACIC/government over informal content).
