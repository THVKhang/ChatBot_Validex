import operator
from typing import Annotated, TypedDict
from app.session_manager import SessionManager

class ParsedData(TypedDict, total=False):
    intent: str
    topic: str
    audience: str
    tone: str
    length: str
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
    
    # UI Metadata
    quality_gate_blocked: bool
