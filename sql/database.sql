-- Embedding dimensions must match the configured EMBEDDING_PROVIDER:
--   local  bge-base-finetuned-validex .... 768  (current default)
--   google models/text-embedding-004 ..... 768
--   openai text-embedding-3-small ........ 1536
-- Override without editing this file:  psql -v embedding_dim=1536 -f sql/database.sql
-- app/vector_repository.py:initialize_schema() creates the same shape at runtime
-- and refuses to write when the table's dimension disagrees with the model.
\if :{?embedding_dim}
\else
    \set embedding_dim 768
\endif

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS validex_knowledge (
    chunk_id TEXT PRIMARY KEY,
    chunk_hash TEXT NOT NULL UNIQUE,
    embedding_provider TEXT NOT NULL DEFAULT 'unknown',
    doc_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    topic TEXT NOT NULL,
    region TEXT NOT NULL,
    title TEXT NOT NULL,
    authority_score DOUBLE PRECISION NOT NULL,
    approved BOOLEAN NOT NULL,
    content TEXT NOT NULL,
    embedding vector(:embedding_dim) NOT NULL,
    -- Legal metadata (Trụ Cột 3: Metadata Enrichment)
    status TEXT NOT NULL DEFAULT 'in_force',              -- 'in_force', 'repealed', 'amended'
    jurisdiction TEXT NOT NULL DEFAULT 'Commonwealth',     -- 'Commonwealth', 'NSW', 'VIC', 'QLD', etc.
    document_type TEXT NOT NULL DEFAULT 'webpage',         -- 'legislation', 'regulation', 'guide', 'faq', 'webpage', 'pdf'
    act_name TEXT NOT NULL DEFAULT '',                     -- e.g. 'Crimes Act 1914'
    section_ref TEXT NOT NULL DEFAULT '',                  -- e.g. 'Part VIIC - Section 85ZM'
    effective_date TEXT NOT NULL DEFAULT '',               -- ISO date: '2024-01-01'
    parent_context TEXT NOT NULL DEFAULT '',               -- Breadcrumb: 'Crimes Act 1914 > Part VIIC > Division 3'
    -- Delta Sync tracking
    last_verified_at TIMESTAMPTZ,
    superseded_by TEXT DEFAULT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fts_content tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''))) STORED
);

CREATE INDEX IF NOT EXISTS idx_validex_knowledge_fts ON validex_knowledge USING GIN (fts_content);

CREATE INDEX IF NOT EXISTS idx_validex_knowledge_topic ON validex_knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_source_domain ON validex_knowledge(source_domain);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_provider ON validex_knowledge(embedding_provider);

-- Legal metadata indexes
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_status ON validex_knowledge(status);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_jurisdiction ON validex_knowledge(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_document_type ON validex_knowledge(document_type);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_act_name ON validex_knowledge(act_name) WHERE act_name != '';
-- Composite: the retriever's most common filter pattern
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_status_jurisdiction ON validex_knowledge(status, jurisdiction);


-- HNSW ANN index: better than IVFFlat for small/medium datasets (<100k rows).
-- IVFFlat requires ~lists*30 rows (lists=100 → 3000 rows) to train well and degrades below that.
-- HNSW has no training phase, works well at any dataset size, and provides good recall.
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_embedding_hnsw
ON validex_knowledge
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Optional: switch to IVFFlat only when the table has >50k rows for better throughput.
-- CREATE INDEX IF NOT EXISTS idx_validex_knowledge_embedding_ivfflat
-- ON validex_knowledge
-- USING ivfflat (embedding vector_cosine_ops)
-- WITH (lists = 100);

-- Semantic Cache Table. Dimensions follow the *local* encoder in
-- app/local_semantics.py (bge-base-finetuned-validex = 768, all-MiniLM-L6-v2 =
-- 384), which is not necessarily the knowledge-base encoder. app/semantic_cache.py
-- owns this table at runtime: it verifies the dimension on first use and
-- rebuilds the table if the encoder changed, since vectors from a different
-- model are not comparable and the cache would only ever miss.
CREATE TABLE IF NOT EXISTS validex_semantic_cache (
    id SERIAL PRIMARY KEY,
    prompt_text TEXT NOT NULL,
    prompt_embedding vector(:embedding_dim) NOT NULL,
    generated_response JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_validex_semantic_cache_prompt
ON validex_semantic_cache (md5(prompt_text));

-- Index for semantic cache using HNSW for fast similarity search
CREATE INDEX IF NOT EXISTS idx_validex_semantic_cache_embedding_hnsw
ON validex_semantic_cache
USING hnsw (prompt_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Token Usage Tracking Table
CREATE TABLE IF NOT EXISTS token_usage_log (
    id SERIAL PRIMARY KEY,
    date TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INT NOT NULL DEFAULT 0,
    output_tokens INT NOT NULL DEFAULT 0,
    request_count INT NOT NULL DEFAULT 0,
    error_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
