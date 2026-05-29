import operator
from typing import Annotated, TypedDict
from app.session_manager import SessionManager

class ParsedData(TypedDict, total=False):
    intent: str
    topic: str
    audience: str
    tone: str
    length: str
    language: str
    target_sections: int
    target_images: int
    context_note: str

class RetrievedDoc(TypedDict):
    doc_id: str
    content: str
    score: float
    source: str
    title: str
    source_url: str

class GraphState(TypedDict):
    prompt: str
    session: SessionManager
    request_id: str | None
    
    parsed: ParsedData
    
    # We use list here. Annotated with operator.add is only needed if we want multiple nodes to append to it in parallel.
    retrieved_docs: list[RetrievedDoc]
    
    # Generation outputs
    title: str
    outline: list[str]
    draft: str
    sources_used: list[str]
    
    # Editor feedback
    editor_feedback: str | None
    revision_count: int
    previous_draft: str | None  # Comparison Gate: track draft before revision
    
    # UI Metadata
    quality_gate_blocked: bool
    retrieval_meta: dict
    
    # Agentic Loop Tracking
    retrieval_attempts: int
    tried_queries: list[str]
    rag_feedback: str | None
    loop_step: int
    
    # Supervisor Architecture
    complexity_level: str  # "simple" or "complex"
    supervisor_notes: str

    # Edit Intent: when user wants to modify existing blog (skip RAG)
    edit_instruction: str | None

    # E2E / API flag
    from_api: bool

    # ML/DL Quality Control Pipeline
    ml_features: dict  # Feature vector extracted by ML Data Collector
    ml_quality_prediction: dict | None  # ML model prediction result

