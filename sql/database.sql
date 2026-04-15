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

-- Optional ANN index for faster similarity search on larger datasets.
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_embedding_ivfflat
ON validex_knowledge
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
