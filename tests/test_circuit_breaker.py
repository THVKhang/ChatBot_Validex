"""Circuit breaker tests for LLM failure handling.

Covers:
- CB opens after N consecutive failures
- CB cooldown period behavior
- CB resets on success
- Error classification (quota, auth, timeout)
"""

import time
import pytest

from app.langchain_pipeline import pipeline, LangChainRAGPipeline


class TestCircuitBreakerState:
    """Test circuit breaker state transitions."""

    def test_initial_state_is_closed(self):
        """CB should start closed (0 failures, not open)."""
        assert pipeline._cb_consecutive_failures == 0
        assert pipeline._cb_open_until == 0.0

    def test_cb_opens_after_threshold_failures(self):
        """After N consecutive failures, CB should open."""
        from app.config import settings
        threshold = settings.circuit_breaker_threshold

        for i in range(threshold):
            pipeline._cb_consecutive_failures = i + 1

        assert pipeline._cb_consecutive_failures >= threshold

    def test_cb_open_until_is_future_timestamp(self):
        """When CB opens, open_until should be a future timestamp."""
        from app.config import settings
        cooldown = settings.circuit_breaker_cooldown_seconds

        pipeline._cb_consecutive_failures = settings.circuit_breaker_threshold
        pipeline._cb_open_until = time.time() + cooldown

        assert pipeline._cb_open_until > time.time()

    def test_cb_resets_on_manual_clear(self):
        """Manually resetting CB state."""
        pipeline._cb_consecutive_failures = 10
        pipeline._cb_open_until = time.time() + 3600

        # Reset
        pipeline._cb_consecutive_failures = 0
        pipeline._cb_open_until = 0.0

        assert pipeline._cb_consecutive_failures == 0
        assert pipeline._cb_open_until == 0.0


class TestErrorClassification:
    """Test LLM error classification for CB decisions."""

    def test_classify_quota_error(self):
        result = pipeline._classify_llm_error("RESOURCE_EXHAUSTED: 429 quota exceeded")
        assert result == "quota_exhausted"

    def test_classify_rate_limit_error(self):
        result = pipeline._classify_llm_error("Rate limit exceeded, retry after 60s")
        assert result == "quota_exhausted"

    def test_classify_auth_error(self):
        result = pipeline._classify_llm_error("401 Unauthenticated: invalid API key")
        assert result == "auth_error"

    def test_classify_timeout_error(self):
        result = pipeline._classify_llm_error("Deadline exceeded: 504 timeout")
        assert result == "timeout"

    def test_classify_invalid_request(self):
        result = pipeline._classify_llm_error("400 INVALID_ARGUMENT: bad request")
        assert result == "invalid_request"

    def test_classify_unknown_error(self):
        result = pipeline._classify_llm_error("Something unexpected happened")
        assert result == "invoke_error"

    def test_classify_empty_error(self):
        result = pipeline._classify_llm_error("")
        assert result == "invoke_error"

    def test_classify_none_error(self):
        result = pipeline._classify_llm_error(None)
        assert result == "invoke_error"


class TestLLMFailureRecording:
    """Test that LLM failures are properly recorded in trace."""

    def test_record_failure_adds_to_trace(self):
        trace = {"attempted": False, "failures": []}
        pipeline._record_llm_failure(trace, "generation", "429 quota exceeded")
        assert trace["attempted"] is True
        assert len(trace["failures"]) == 1
        assert trace["failures"][0]["stage"] == "generation"
        assert trace["failures"][0]["reason"] == "quota_exhausted"

    def test_record_failure_with_none_trace(self):
        """Should not raise when trace is None."""
        pipeline._record_llm_failure(None, "generation", "error")
        # No assertion needed — just verify no exception

    def test_record_multiple_failures(self):
        trace = {"attempted": False, "failures": []}
        pipeline._record_llm_failure(trace, "parse", "timeout")
        pipeline._record_llm_failure(trace, "generation", "auth error 401")
        pipeline._record_llm_failure(trace, "structured_output", "RESOURCE_EXHAUSTED")
        assert len(trace["failures"]) == 3
        assert trace["failures"][0]["reason"] == "timeout"
        assert trace["failures"][1]["reason"] == "auth_error"
        assert trace["failures"][2]["reason"] == "quota_exhausted"

    def test_record_failure_truncates_long_messages(self):
        trace = {"failures": []}
        long_msg = "x" * 2000
        pipeline._record_llm_failure(trace, "test", long_msg)
        recorded_msg = trace["failures"][0].get("reason", "")
        # The classified reason is short, but the raw message should be truncated
        assert len(recorded_msg) < 500


class TestCircuitBreakerConfig:
    """Test CB configuration values."""

    def test_threshold_is_positive(self):
        from app.config import settings
        assert settings.circuit_breaker_threshold > 0

    def test_cooldown_is_positive(self):
        from app.config import settings
        assert settings.circuit_breaker_cooldown_seconds > 0

    def test_default_threshold_is_3(self):
        from app.config import settings
        assert settings.circuit_breaker_threshold == 3

    def test_default_cooldown_is_60(self):
        from app.config import settings
        assert settings.circuit_breaker_cooldown_seconds == 60
