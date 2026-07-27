-- ============================================================
-- Legal Metadata Migration
-- Safe to run on existing database (all ADD COLUMN IF NOT EXISTS)
-- ============================================================

-- Core legal metadata columns
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'in_force';
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS jurisdiction TEXT NOT NULL DEFAULT 'Commonwealth';
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS document_type TEXT NOT NULL DEFAULT 'webpage';
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS act_name TEXT NOT NULL DEFAULT '';
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS section_ref TEXT NOT NULL DEFAULT '';
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS effective_date TEXT NOT NULL DEFAULT '';
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS parent_context TEXT NOT NULL DEFAULT '';

-- Metadata filtering indexes (critical for retriever performance)
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_status ON validex_knowledge(status);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_jurisdiction ON validex_knowledge(jurisdiction);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_document_type ON validex_knowledge(document_type);
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_act_name ON validex_knowledge(act_name) WHERE act_name != '';

-- Composite index for the most common retriever filter pattern:
-- WHERE status = 'in_force' AND jurisdiction = ?
CREATE INDEX IF NOT EXISTS idx_validex_knowledge_status_jurisdiction
ON validex_knowledge(status, jurisdiction);

-- Delta Sync: track when each chunk was last verified against source
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMPTZ;
ALTER TABLE validex_knowledge ADD COLUMN IF NOT EXISTS superseded_by TEXT DEFAULT NULL;

-- ============================================================
-- Data Changelog (for Delta Sync audit trail)
-- ============================================================
CREATE TABLE IF NOT EXISTS validex_changelog (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    action TEXT NOT NULL,          -- 'insert', 'update', 'repeal', 'delete'
    old_status TEXT,
    new_status TEXT,
    source_url TEXT,
    reason TEXT DEFAULT '',
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validex_changelog_chunk_id ON validex_changelog(chunk_id);
CREATE INDEX IF NOT EXISTS idx_validex_changelog_action ON validex_changelog(action);
