"""Graph state machine tests for the multi-agent LangGraph pipeline.

Covers:
- Route decisions (simple vs complex topics)
- Edit intent skips RAG
- Editor revision loop limits (circuit breaker at 3)
- Supervisor complexity scoring
- Deep Researcher deduplication
"""

import pytest
from langgraph.graph import END

from app.graph import (
    COMPLEX_INDICATORS,
    SupervisorNode,
    DeepResearcherNode,
    route_after_parser,
    route_after_rag_complex,
    route_after_supervisor,
    route_after_editor,
)


class TestRouteAfterParser:
    """Test conditional routing from Parser node."""

    def test_create_intent_routes_to_researcher(self):
        state = {"edit_instruction": None}
        assert route_after_parser(state) == "Researcher"

    def test_edit_intent_routes_to_writer(self):
        state = {"edit_instruction": "Make it shorter and more professional"}
        assert route_after_parser(state) == "Writer"

    def test_empty_edit_instruction_routes_to_researcher(self):
        state = {"edit_instruction": ""}
        assert route_after_parser(state) == "Researcher"

    def test_missing_edit_instruction_routes_to_researcher(self):
        state = {}
        assert route_after_parser(state) == "Researcher"




class TestRouteAfterEditor:
    """Test Editor exit routing (accept -> END, reject -> Writer)."""

    def test_no_feedback_finishes(self):
        state = {"editor_feedback": None}
        assert route_after_editor(state) == END

    def test_feedback_below_limit_goes_to_writer(self):
        state = {"editor_feedback": "Improve tone", "loop_step": 1, "revision_count": 1}
        assert route_after_editor(state) == "Writer"

    def test_feedback_at_limit_finishes(self):
        """Circuit breaker: stop revising and publish rather than loop forever."""
        state = {"editor_feedback": "Still bad", "loop_step": 3, "revision_count": 3}
        assert route_after_editor(state) == END




class TestRouteAfterRAGComplex:
    """Test RAG → Supervisor routing for complex pipeline."""

    def test_no_feedback_routes_to_supervisor(self):
        state = {"rag_feedback": None, "retrieval_attempts": 0}
        assert route_after_rag_complex(state) == "Supervisor"

    def test_feedback_routes_to_researcher(self):
        state = {"rag_feedback": "Gaps found", "retrieval_attempts": 1}
        assert route_after_rag_complex(state) == "Researcher"

    def test_max_attempts_routes_to_supervisor(self):
        state = {"rag_feedback": "Still gaps", "retrieval_attempts": 2}
        assert route_after_rag_complex(state) == "Supervisor"


class TestRouteAfterSupervisor:
    """Test Supervisor → Writer vs Deep Researcher routing."""

    def test_simple_topic_routes_to_writer(self):
        state = {"complexity_level": "simple"}
        assert route_after_supervisor(state) == "Writer"

    def test_complex_topic_routes_to_deep_researcher(self):
        state = {"complexity_level": "complex"}
        assert route_after_supervisor(state) == "Deep_Researcher"

    def test_missing_complexity_defaults_to_simple(self):
        state = {}
        assert route_after_supervisor(state) == "Writer"


class TestSupervisorNode:
    """Test Supervisor complexity scoring."""

    def test_simple_topic_scores_low(self):
        node = SupervisorNode()
        state = {"parsed": {"topic": "what is a police check", "length": "medium"}}
        result = node.execute(state)
        assert result["complexity_level"] == "simple"

    def test_legal_topic_scores_high(self):
        node = SupervisorNode()
        state = {"parsed": {"topic": "spent convictions legislation compliance", "length": "long"}}
        result = node.execute(state)
        assert result["complexity_level"] == "complex"

    def test_comparison_topic_scores_high(self):
        node = SupervisorNode()
        state = {"parsed": {"topic": "NSW versus Victoria privacy legislation", "length": "medium"}}
        result = node.execute(state)
        assert result["complexity_level"] == "complex"

    def test_empty_topic_scores_simple(self):
        node = SupervisorNode()
        state = {"parsed": {"topic": "", "length": "short"}}
        result = node.execute(state)
        assert result["complexity_level"] == "simple"

    def test_complex_indicators_list_is_populated(self):
        assert len(COMPLEX_INDICATORS) > 10
        assert "legislation" in COMPLEX_INDICATORS
        assert "compliance" in COMPLEX_INDICATORS


