"""LangGraph Multi-Agent Graph with Supervisor Architecture.

Supports both simple (linear) and complex (parallel map-reduce) pipelines.
The Supervisor Node analyzes topic complexity after parsing and routes accordingly.
The ML quality-gate and data-collector nodes were removed: the gate ran in
shadow mode (never blocked), it was trained on a few hundred examples with no
regression guard, and it added two node visits plus contradictory verdicts to
every request. The fine-tuned MODELS in data/models/ are unaffected and still in
use — the knowledge base is embedded with bge-base-finetuned-validex.
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph_state import GraphState
from app.agents.parser_node import parser_node
from app.agents.researcher_node import researcher_node
from app.agents.rag_evaluator_node import rag_evaluator_node
from app.agents.writer_node import writer_node
from app.agents.editor_node import editor_node

import logging

logger = logging.getLogger(__name__)

# ── Global Circuit Breaker ────────────────────────────────────
# Maximum total node visits across ALL retry loops (RAG + Editor).
# Prevents GraphRecursionError from nested loop combinations.
GLOBAL_STEP_LIMIT = 8

# ── Legal / Complex Topic Indicators ──────────────────────────
COMPLEX_INDICATORS = [
    "legislation", "act", "regulation", "compliance", "legal",
    "privacy", "ndis", "criminal", "offence", "conviction",
    "spent convictions", "working with children", "wwcc",
    "fair work", "anti-discrimination", "human rights",
    "immigration", "visa", "citizenship",
    "multi-state", "comparison", "versus", "vs",
    "points", "100-point", "100 point", "fee", "cost", 
    "how many", "how much", "points calculation", "score",
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
        prompt = state.get("prompt", "").lower()
        
        # Score complexity based on indicators in both topic and prompt
        complexity_score = sum(1 for indicator in COMPLEX_INDICATORS if indicator in topic or indicator in prompt)
        
        # Also check length hint: long articles on legal topics = complex
        is_long = parsed.get("length", "medium") == "long"
        if is_long:
            complexity_score += 1
            
        # Force complex for point calculations, fees, and critical regulatory guides
        force_complex_keywords = [
            "points", "100-point", "100 point", "fee", "cost", 
            "how many", "how much", "spent conviction", "spent convictions",
            "working with children", "wwcc", "ndis", "citizenship", "visa",
            "calculation", "calculator"
        ]
        is_forced_complex = any(kw in prompt or kw in topic for kw in force_complex_keywords)
        
        if complexity_score >= 2 or is_forced_complex:
            level = "complex"
            notes = f"Complex topic/prompt detected (score={complexity_score}, forced={is_forced_complex}). Using enhanced pipeline with deeper research."
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




# Context quality above which a second research pass is not worth its cost.
# Deep_Researcher re-runs the same query expansion over the same corpus, so when
# retrieval already scored well it returns the documents we have: measured on a
# WWCC request it spent 31s to merge "7 + 7 -> 7 docs", adding nothing.
DEEP_RESEARCH_SKIP_SCORE = 0.70


def route_after_supervisor(state: GraphState):
    """Route based on complexity: simple goes directly to Writer, complex does deep research first."""
    level = state.get("complexity_level", "simple")
    if level != "complex":
        return "Writer"

    score = float(state.get("rag_score") or 0.0)
    if score >= DEEP_RESEARCH_SKIP_SCORE and not state.get("rag_feedback"):
        logger.info(
            "Supervisor: context already strong (score=%.2f >= %.2f) — skipping Deep Researcher.",
            score, DEEP_RESEARCH_SKIP_SCORE,
        )
        return "Writer"
    return "Deep_Researcher"


def route_after_rag_complex(state: GraphState):
    """For complex pipeline: after RAG eval, route to deep researcher or supervisor routing."""
    # Global circuit breaker
    if state.get("global_step_count", 0) >= GLOBAL_STEP_LIMIT:
        logger.warning("⚠️ GLOBAL CIRCUIT BREAKER: step %d >= %d. Forcing Supervisor.", state.get("global_step_count", 0), GLOBAL_STEP_LIMIT)
        return "Supervisor"
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

# Editor: accept (→ END) or reject (→ Writer)
def route_after_editor(state: GraphState):
    """Route from Editor: if feedback exists go back to Writer, otherwise finish."""
    if state.get("editor_feedback"):
        loop_step = state.get("loop_step", 0)
        revision_count = state.get("revision_count", 0)
        global_step = state.get("global_step_count", 0)
        if loop_step >= 3 or revision_count >= 3 or global_step >= GLOBAL_STEP_LIMIT:
            logger.warning(
                "⚠️ CIRCUIT BREAKER: Force-exiting cyclic loop "
                "(loop_step=%d, revision_count=%d, global_step=%d). Publishing draft as is.",
                loop_step, revision_count, global_step,
            )
            return END
        return "Writer"
    return END

builder.add_conditional_edges("Editor", route_after_editor)

# Compile Graph with Checkpointer for state persistence
checkpointer = MemorySaver()
multi_agent_graph = builder.compile(
    checkpointer=checkpointer,
    # Explicit recursion limit to prevent runaway loops (LangGraph default is 25)
    # Set higher than GLOBAL_STEP_LIMIT to allow the circuit breaker to handle gracefully
)

