"""LangGraph Multi-Agent Graph with Supervisor Architecture.

Supports both simple (linear) and complex (parallel map-reduce) pipelines.
The Supervisor Node analyzes topic complexity after parsing and routes accordingly.
Includes ML/DL Quality Control pipeline integration for training data collection.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph_state import GraphState
from app.agents.parser_node import parser_node
from app.agents.researcher_node import researcher_node
from app.agents.rag_evaluator_node import rag_evaluator_node
from app.agents.writer_node import writer_node
from app.agents.editor_node import editor_node
from app.agents.ml_collector_node import ml_collector_node
from app.agents.ml_gate_node import ml_gate_node

import logging

logger = logging.getLogger(__name__)

# ── Legal / Complex Topic Indicators ──────────────────────────
COMPLEX_INDICATORS = [
    "legislation", "act", "regulation", "compliance", "legal",
    "privacy", "ndis", "criminal", "offence", "conviction",
    "spent convictions", "working with children", "wwcc",
    "fair work", "anti-discrimination", "human rights",
    "immigration", "visa", "citizenship",
    "multi-state", "comparison", "versus", "vs",
]


from app.agents.base import BaseAgentNode

# ── Supervisor Node ──────────────────────────────────────────
class SupervisorNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Analyzes parsed topic and determines pipeline complexity.
        
        Routes to either:
        - 'simple': Standard linear pipeline (Researcher → Writer → Editor)
        - 'complex': Enhanced pipeline with deeper research iterations
        """
        parsed = state.get("parsed", {})
        topic = parsed.get("topic", "").lower()
        
        # Score complexity based on indicators
        complexity_score = sum(1 for indicator in COMPLEX_INDICATORS if indicator in topic)
        
        # Also check length hint: long articles on legal topics = complex
        is_long = parsed.get("length", "medium") == "long"
        if is_long:
            complexity_score += 1
        
        if complexity_score >= 2:
            level = "complex"
            notes = f"Complex topic detected (score={complexity_score}). Using enhanced pipeline with deeper research."
        else:
            level = "simple"
            notes = f"Standard topic (score={complexity_score}). Using optimized linear pipeline."
        
        logger.info(f"Supervisor: {notes}")
        
        return {
            "complexity_level": level,
            "supervisor_notes": notes,
        }

supervisor_node = SupervisorNode()


# ── Deep Researcher Node (for complex topics) ───────────────
class DeepResearcherNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Extended researcher that performs additional multi-query retrieval 
        for complex legal/compliance topics. Runs after initial researcher 
        to fill gaps identified by RAG evaluator."""
        logger.info("Executing Deep Researcher Node (complex pipeline)")
        
        # Re-run researcher with augmented queries
        parsed = state.get("parsed", {})
        topic = parsed.get("topic", "")
        
        # Generate additional legal-specific queries
        supplementary_queries = [
            f"{topic} Australian legislation requirements",
            f"{topic} compliance obligations employer",
            f"{topic} recent changes updates 2024 2025",
        ]
        
        # Use existing researcher logic but with expanded tried_queries
        existing_tried = state.get("tried_queries", [])
        new_tried = existing_tried + supplementary_queries
        
        # Call the standard researcher with augmented state
        augmented_state = dict(state)
        augmented_state["tried_queries"] = new_tried
        augmented_state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
        
        result = researcher_node(augmented_state)
        
        # Merge new docs with existing ones
        existing_docs = state.get("retrieved_docs", [])
        new_docs = result.get("retrieved_docs", [])
        
        # Deduplicate by doc_id
        seen_ids = {d["doc_id"] for d in existing_docs}
        merged = list(existing_docs)
        for doc in new_docs:
            if doc["doc_id"] not in seen_ids:
                merged.append(doc)
                seen_ids.add(doc["doc_id"])
        
        # Cap merged documents at 7 to prevent context stuffing
        merged = merged[:7]
        
        logger.info(f"Deep Researcher: merged {len(existing_docs)} + {len(new_docs)} → {len(merged)} docs")
        
        return {
            "retrieved_docs": merged,
            "tried_queries": new_tried,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

deep_researcher_node = DeepResearcherNode()



# ── Routing Functions ────────────────────────────────────────
def route_after_parser(state: GraphState):
    """Conditional edge from Parser: If editing existing blog, skip RAG. Else normal flow."""
    edit_instruction = state.get("edit_instruction")
    if edit_instruction:
        logger.info(f"Router: Edit intent detected ('{edit_instruction[:50]}...') → skipping to Writer")
        return "Writer"
    return "Researcher"


def route_after_editor(state: GraphState):
    """Conditional edge from Editor: If feedback exists, go back to Writer. Else END."""
    if state.get("editor_feedback"):
        # Circuit Breaker: limit revision cycles
        loop_step = state.get("loop_step", 0)
        revision_count = state.get("revision_count", 0)
        if loop_step >= 3 or revision_count >= 3:
            logger.warning(
                f"⚠️ CIRCUIT BREAKER: Force-exiting cyclic loop after "
                f"loop_step={loop_step}, revision_count={revision_count}. Publishing draft as is."
            )
            return END
        return "Writer"
    return END


def route_after_rag(state: GraphState):
    """Conditional edge from RAG Evaluator: If feedback exists and attempts < 2, go back to Researcher. Else route by complexity."""
    attempts = state.get("retrieval_attempts", 0)
    if state.get("rag_feedback") and attempts < 2:
        return "Researcher"
    return "Writer"


def route_after_supervisor(state: GraphState):
    """Route based on complexity: simple goes directly to Writer, complex does deep research first."""
    level = state.get("complexity_level", "simple")
    if level == "complex":
        return "Deep_Researcher"
    return "Writer"


def route_after_rag_complex(state: GraphState):
    """For complex pipeline: after RAG eval, route to deep researcher or supervisor routing."""
    attempts = state.get("retrieval_attempts", 0)
    if state.get("rag_feedback") and attempts < 2:
        return "Researcher"
    return "Supervisor"


# ── Build Graph ──────────────────────────────────────────────
builder = StateGraph(GraphState)

# Add Nodes
builder.add_node("Parser", parser_node)
builder.add_node("Researcher", researcher_node)
builder.add_node("RAG_Evaluator", rag_evaluator_node)
builder.add_node("Supervisor", supervisor_node)
builder.add_node("Deep_Researcher", deep_researcher_node)
builder.add_node("Writer", writer_node)
builder.add_node("Editor", editor_node)
builder.add_node("ML_Quality_Gate", ml_gate_node)
builder.add_node("ML_Collector", ml_collector_node)

# Set Entry Point
builder.set_entry_point("Parser")

# Parser: edit → Writer (skip RAG), create → Researcher (normal flow)
builder.add_conditional_edges("Parser", route_after_parser)

# Linear flow: Researcher → RAG_Evaluator → Supervisor
builder.add_edge("Researcher", "RAG_Evaluator")

# RAG Evaluator: retry or proceed to Supervisor
builder.add_conditional_edges("RAG_Evaluator", route_after_rag_complex)

# Supervisor decides: simple (→ Writer) or complex (→ Deep_Researcher)
builder.add_conditional_edges("Supervisor", route_after_supervisor)

# Deep Researcher feeds into Writer
builder.add_edge("Deep_Researcher", "Writer")

# Writer → Editor
builder.add_edge("Writer", "Editor")

# Editor: accept (→ ML_Quality_Gate) or reject (→ Writer)
def route_after_editor_with_ml(state: GraphState):
    """Route from Editor: If feedback exists, go back to Writer. Else ML Quality Gate."""
    if state.get("editor_feedback"):
        loop_step = state.get("loop_step", 0)
        revision_count = state.get("revision_count", 0)
        if loop_step >= 3 or revision_count >= 3:
            logger.warning(
                f"⚠️ CIRCUIT BREAKER: Force-exiting cyclic loop after "
                f"loop_step={loop_step}, revision_count={revision_count}. Publishing draft as is."
            )
            return "ML_Quality_Gate"
        return "Writer"
    return "ML_Quality_Gate"

builder.add_conditional_edges("Editor", route_after_editor_with_ml)

# ML Quality Gate: dual-scope routing
#   - pass → ML_Collector (normal flow)
#   - block (retrieval issue) → Researcher (fetch more docs)
#   - block (generation issue) → Writer (rewrite draft)
def route_after_ml_gate(state: GraphState):
    """Route from ML Gate based on dual-scope failure diagnosis.

    In SHADOW mode: always passes to ML_Collector (never blocks).
    In ENFORCE mode: routes to Researcher or Writer based on failure source.
    """
    prediction = state.get("ml_quality_prediction") or {}
    mode = prediction.get("gate_mode", "shadow")

    # Shadow mode: never block, always pass through
    if mode == "shadow":
        return "ML_Collector"

    # Enforce mode: check if ML Gate set blocking feedback
    if prediction.get("model_available", False):
        quality = prediction.get("quality_class", "medium")
        confidence = prediction.get("quality_confidence", 0.0)
        failure_source = prediction.get("failure_source", "none")
        revision_count = state.get("revision_count", 0)
        retrieval_attempts = state.get("retrieval_attempts", 0)

        # Only block when confident AND within retry limits
        if quality == "low" and confidence >= 0.80 and revision_count < 2:
            if failure_source == "retrieval" and retrieval_attempts < 2:
                logger.warning(
                    "ML Gate [ENFORCE] → Researcher (retrieval weak, quality=%s, conf=%.2f)",
                    quality, confidence,
                )
                return "Researcher"
            else:
                logger.warning(
                    "ML Gate [ENFORCE] → Writer (generation weak, quality=%s, conf=%.2f)",
                    quality, confidence,
                )
                return "Writer"

    return "ML_Collector"

builder.add_conditional_edges("ML_Quality_Gate", route_after_ml_gate)

# ML Collector → END (passive data collection, never blocks)
builder.add_edge("ML_Collector", END)

# Compile Graph with Checkpointer for state persistence
checkpointer = MemorySaver()
multi_agent_graph = builder.compile(checkpointer=checkpointer)

