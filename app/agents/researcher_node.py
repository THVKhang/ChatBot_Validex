import logging
from app.graph_state import GraphState, RetrievedDoc
from app.langchain_pipeline import pipeline
from app.config import settings

logger = logging.getLogger(__name__)

def researcher_node(state: GraphState) -> GraphState:
    """Agent that handles vector search, reranking, and web search fallback."""
    logger.info("Executing Researcher Node")
    
    parsed = state["parsed"]
    topic = parsed.get("topic", "")
    
    # Check if there is an uploaded document in the session that should override RAG
    session = state["session"]
    last_turn = session.latest_turn()
    uploaded_text = getattr(last_turn, "uploaded_file_content", None) if last_turn else None
    
    if uploaded_text:
        # Use uploaded file as the only source
        docs = [
            RetrievedDoc(
                doc_id="uploaded_file",
                content=uploaded_text,
                score=100.0,
                source="User Upload"
            )
        ]
        return {"retrieved_docs": docs}
        
    # Standard Retrieval Process
    payload = {
        "effective_topic": topic,
        "retrieval_top_k": settings.top_k,
    }
    
    try:
        bundle = pipeline._retrieve(payload)
        docs = []
        for d in bundle.documents:
            docs.append(RetrievedDoc(
                doc_id=d.metadata.get("doc_id", "unknown"),
                content=d.page_content,
                score=d.metadata.get("score", 0.0),
                source=d.metadata.get("source", "Internal Database")
            ))
        return {"retrieved_docs": docs}
    except Exception as exc:
        logger.warning(f"Researcher node failed: {exc}")
        return {"retrieved_docs": []}
