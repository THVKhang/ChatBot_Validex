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
  title: string;
  outline: string[];
  draft: string;
  sources_used: string[];
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
