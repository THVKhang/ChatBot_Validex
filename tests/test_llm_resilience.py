"""Tests for LLM error classification and response parsing.

These cover failures that shipped to production undetected:
  - Gemini 3.x returns `.content` as a LIST of blocks; `str(...)` on it published
    the Python repr (`[{'type': 'text', ...}]`) as the article body.
  - A retired/unauthorised model was retried 3x with backoff on every call,
    costing 54 wasted requests and ~54s of sleep per article.
  - A per-day quota was treated as transient, turning one article into a
    33-minute grind instead of a fast, clear failure.
"""

import pytest

from app.llm.provider import (
    _is_daily_quota_exhausted,
    _is_permanent_error,
    extract_text,
    is_model_dead,
    is_quota_exhausted,
    reset_dead_models,
    reset_quota_state,
)


@pytest.fixture(autouse=True)
def _clean_provider_state():
    reset_dead_models()
    reset_quota_state()
    yield
    reset_dead_models()
    reset_quota_state()


class _Response:
    def __init__(self, content):
        self.content = content


class TestExtractText:
    def test_plain_string_content(self):
        assert extract_text(_Response("# Title\n\nbody")) == "# Title\n\nbody"

    def test_gemini_3x_content_blocks(self):
        """The exact shape that used to be published as a Python repr."""
        blocks = [{"type": "text", "text": "# Title"}, {"type": "text", "text": "\n\nbody"}]
        assert extract_text(_Response(blocks)) == "# Title\n\nbody"

    def test_skips_non_text_blocks(self):
        blocks = [
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "text", "text": "visible answer"},
        ]
        assert extract_text(_Response(blocks)) == "visible answer"

    def test_list_of_bare_strings(self):
        assert extract_text(_Response(["a", "b"])) == "ab"

    def test_empty_and_none(self):
        assert extract_text(_Response("")) == ""
        assert extract_text(None) == ""

    def test_never_returns_python_repr(self):
        """Regression guard: output must not look like a serialised structure."""
        blocks = [{"type": "text", "text": "Employers must verify identity."}]
        result = extract_text(_Response(blocks))
        assert not result.startswith("[{")
        assert "'type':" not in result


class TestPermanentErrors:
    @pytest.mark.parametrize("message", [
        "Error code: 404 - {'message': 'The model `x` does not exist', 'code': 'model_not_found'}",
        "404 NOT_FOUND. This model models/gemini-2.0-flash-lite is no longer available.",
        "Error code: 401 - invalid_api_key",
    ])
    def test_permanent_errors_detected(self, message):
        assert _is_permanent_error(Exception(message)) is True

    @pytest.mark.parametrize("message", [
        "429 RESOURCE_EXHAUSTED. Please retry in 14.59s.",
        "503 UNAVAILABLE. This model is currently experiencing high demand.",
        "Connection reset by peer",
    ])
    def test_transient_errors_are_not_permanent(self, message):
        assert _is_permanent_error(Exception(message)) is False


class TestQuotaClassification:
    def test_per_day_quota_is_parked(self):
        err = Exception(
            "429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
            "generate_content_free_tier_requests, quotaId: "
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20"
        )
        assert _is_daily_quota_exhausted(err) is True
        # A day-long limit must not be mistaken for a broken model.
        assert _is_permanent_error(err) is False

    def test_per_minute_throttle_is_retried_not_parked(self):
        err = Exception(
            "429 RESOURCE_EXHAUSTED. Please retry in 14.59s. "
            "quotaId: GenerateRequestsPerMinutePerProject"
        )
        assert _is_daily_quota_exhausted(err) is False

    def test_non_quota_error_is_not_quota(self):
        assert _is_daily_quota_exhausted(Exception("503 UNAVAILABLE")) is False


class TestModelStateTracking:
    def test_dead_and_quota_state_start_clean(self):
        assert is_model_dead("models/anything") is False
        assert is_quota_exhausted("models/anything") is False
