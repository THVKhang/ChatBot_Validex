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
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validex_knowledge_topic ON validex_knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_source_domain ON validex_knowledge(source_domain);

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
