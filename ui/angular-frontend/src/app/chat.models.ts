export interface ParsedResult {
  intent: string;
  topic: string;
  audience: string;
  tone: string;
  length: string;
  context_note: string;
}

export interface RetrievedResult {
  doc_id: string;
  score: number;
  semantic_score: number;
  snippet: string;
}

export interface RetrievalMeta {
  status: string;
  confidence: number;
  top_score: number;
  reason: string;
}

export interface GeneratedResult {
  sections?: BlogSection[];
  title: string;
  outline: string[];
  draft: string;
  sources_used: string[];
}

export interface BlogSection {
  heading: string;
  body: string;
  image_url: string;
  image_alt: string;
}

export interface ChatApiResponse {
  session_id: string;
  parsed: ParsedResult;
  retrieved: RetrievedResult[];
  retrieval_meta: RetrievalMeta;
  generated: GeneratedResult;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  text: string;
  payload?: ChatApiResponse;
}

export interface ReportSummary {
  id: string;
  created_at: string;
  session_id: string | null;
  title: string;
  prompt: string;
  draft_preview: string;
}

export interface ReportDetail {
  id: string;
  created_at: string;
  session_id: string | null;
  title: string;
  prompt: string;
  outline: string[];
  draft: string;
  sources_used: string[];
  sections?: BlogSection[];
}

export interface SaveReportRequest {
  prompt: string;
  session_id: string | null;
  generated: {
    title: string;
    outline: string[];
    draft: string;
    sources_used: string[];
    sections?: BlogSection[];
  };
}

export interface ReportsListResponse {
  reports: ReportSummary[];
}

export interface ReportDetailResponse {
  report: ReportDetail;
}

export interface SaveReportResponse {
  report: ReportDetail;
}

export interface DeleteReportResponse {
  status: 'deleted';
  report_id: string;
}

export interface HealthRuntime {
  retrieval_mode: string;
  generation_mode: string;
  quality_gate_enabled: boolean;
  quality_rules: {
    min_sections: number;
    min_sources_used: number;
    min_draft_chars: number;
  };
}

export interface HealthResponse {
  status: string;
  runtime: HealthRuntime;
  flags: {
    use_live_llm: boolean;
    use_pinecone_retrieval: boolean;
    use_agentic_rag: boolean;
    use_rate_limit: boolean;
    use_redis_rate_limit?: boolean;
  };
}

export interface MetricsResponse {
  chat_requests_total: number;
  chat_errors_total: number;
  quality_gate_blocked_total: number;
  generation_mode_count: Record<string, number>;
  retrieval_mode_count: Record<string, number>;
  latency: {
    samples: number;
    avg_ms: number;
    p95_ms: number;
  };
}
