-- Validex AI: MLOps and A/B Testing Tables

CREATE TABLE IF NOT EXISTS prompt_ab_tests (
    run_id UUID PRIMARY KEY,
    session_id TEXT,
    prompt_version TEXT NOT NULL,
    topic TEXT NOT NULL,
    editor_verdict TEXT NOT NULL,          -- e.g., 'ACCEPTED', 'REJECTED', 'LLM_PASSED'
    structural_issues JSONB DEFAULT '[]'::jsonb, -- e.g., ['E01:too-short', 'E04:insufficient-headings']
    total_tokens_used INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prompt_ab_tests_version ON prompt_ab_tests(prompt_version);
CREATE INDEX IF NOT EXISTS idx_prompt_ab_tests_verdict ON prompt_ab_tests(editor_verdict);
CREATE INDEX IF NOT EXISTS idx_prompt_ab_tests_created_at ON prompt_ab_tests(created_at);
