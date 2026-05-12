"""Resilient LLM Provider — wraps LangChain LLMs with smart retry, budget-aware
routing, and automatic fallback.

When the primary model (e.g., Groq Llama 3.3 70B) hits a rate limit or exhausts
its daily token budget, the wrapper transparently routes the request to the
fallback model (e.g., Gemini Flash) without any caller-side changes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.llm.token_tracker import token_tracker

logger = logging.getLogger(__name__)


class ResilientLLM:
    """Drop-in wrapper for LangChain ChatModel with resilience features.

    Features:
        - Exponential backoff retry (up to 3 attempts)
        - Pre-flight token budget check (route to fallback before hitting limits)
        - Automatic fallback on any API error (429, 500, timeout, etc.)
        - Token usage tracking per model
    """

    def __init__(
        self,
        primary: Any,
        fallback: Any | None = None,
        primary_model_name: str = "unknown",
        fallback_model_name: str = "unknown",
        budget_threshold: float = 0.85,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_model_name = primary_model_name
        self.fallback_model_name = fallback_model_name
        self.budget_threshold = budget_threshold
        self._active_model: str = primary_model_name

    @property
    def active_model(self) -> str:
        """The model that was used for the most recent invocation."""
        return self._active_model

    def _estimate_tokens(self, input_data: Any) -> int:
        """Rough token estimate from input (4 chars ≈ 1 token)."""
        if isinstance(input_data, str):
            return max(1, len(input_data) // 4)
        if isinstance(input_data, list):
            total_chars = sum(
                len(getattr(msg, "content", str(msg)))
                for msg in input_data
            )
            return max(1, total_chars // 4)
        return 500  # default estimate

    def invoke(self, input_data: Any, **kwargs: Any) -> Any:
        """Invoke with full resilience: budget check → retry → fallback."""
        estimated_input = self._estimate_tokens(input_data)

        # 1. Pre-flight budget check — route to fallback BEFORE hitting limits
        if token_tracker.should_use_fallback(self.primary_model_name, self.budget_threshold):
            if self.fallback:
                logger.warning(
                    "Budget threshold (%.0f%%) reached for %s, routing to fallback %s",
                    self.budget_threshold * 100,
                    self.primary_model_name,
                    self.fallback_model_name,
                )
                return self._invoke_with_tracking(
                    self.fallback, self.fallback_model_name, input_data, estimated_input, **kwargs
                )
            # No fallback available, try primary anyway
            logger.warning("Budget threshold reached but no fallback configured, trying primary")

        # 2. Try primary with exponential backoff
        last_error = None
        for attempt in range(3):
            try:
                result = self._invoke_with_tracking(
                    self.primary, self.primary_model_name, input_data, estimated_input, **kwargs
                )
                return result
            except Exception as exc:
                last_error = exc
                error_str = str(exc).lower()
                is_rate_limit = "429" in error_str or "rate_limit" in error_str or "quota" in error_str

                logger.warning(
                    "Primary LLM (%s) failed (attempt %d/3): %s",
                    self.primary_model_name, attempt + 1, str(exc)[:200],
                )

                # Rate limit / quota → immediately try fallback (no point retrying)
                if is_rate_limit and self.fallback:
                    logger.info("Rate limit detected, immediately routing to fallback %s", self.fallback_model_name)
                    try:
                        return self._invoke_with_tracking(
                            self.fallback, self.fallback_model_name, input_data, estimated_input, **kwargs
                        )
                    except Exception as fb_exc:
                        logger.error("Fallback also failed: %s", fb_exc)
                        raise fb_exc from exc

                # Other errors → exponential backoff then retry
                if attempt < 2:
                    sleep_time = 2 ** attempt
                    logger.info("Retrying in %ds...", sleep_time)
                    time.sleep(sleep_time)

        # 3. All primary retries exhausted → last-resort fallback
        if self.fallback:
            logger.warning("All primary retries exhausted, last-resort fallback to %s", self.fallback_model_name)
            try:
                return self._invoke_with_tracking(
                    self.fallback, self.fallback_model_name, input_data, estimated_input, **kwargs
                )
            except Exception as fb_exc:
                logger.error("Last-resort fallback also failed: %s", fb_exc)
                raise fb_exc from last_error

        raise last_error  # type: ignore[misc]

    def _invoke_with_tracking(
        self,
        llm: Any,
        model_name: str,
        input_data: Any,
        estimated_input: int,
        **kwargs: Any,
    ) -> Any:
        """Invoke an LLM and track token usage."""
        self._active_model = model_name
        try:
            result = llm.invoke(input_data, **kwargs)

            # Estimate output tokens
            output_text = getattr(result, "content", str(result))
            estimated_output = max(1, len(output_text) // 4)

            # Track usage
            token_tracker.record_usage(
                model=model_name,
                input_tokens=estimated_input,
                output_tokens=estimated_output,
            )

            logger.debug(
                "LLM invocation: model=%s, est_input=%d, est_output=%d",
                model_name, estimated_input, estimated_output,
            )
            return result
        except Exception as exc:
            # Track the error
            token_tracker.record_usage(
                model=model_name,
                input_tokens=estimated_input,
                output_tokens=0,
                is_error=True,
            )
            raise


def build_resilient_llm(settings: Any) -> ResilientLLM | None:
    """Factory function that builds a ResilientLLM from app settings.

    Initializes both primary (Groq/OpenAI) and fallback (Gemini) models,
    then wraps them in the resilience layer.
    """
    if not settings.use_live_llm:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        ChatOpenAI = None  # type: ignore[misc, assignment]

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        ChatGoogleGenerativeAI = None  # type: ignore[misc, assignment]

    primary_llm = None
    fallback_llm = None
    primary_name = settings.model_name or "unknown"
    fallback_name = "unknown"

    # 1. Initialize Primary (Groq/OpenAI)
    if settings.openai_api_key and ChatOpenAI is not None:
        try:
            openai_kwargs = {
                "model": settings.model_name,
                "api_key": settings.openai_api_key,
                "temperature": 0.2,
                "max_retries": 0,  # We handle retries ourselves
                "timeout": 30.0,
            }
            if hasattr(settings, "openai_base_url") and settings.openai_base_url:
                openai_kwargs["base_url"] = settings.openai_base_url
                
            primary_llm = ChatOpenAI(**openai_kwargs)
            primary_name = settings.model_name
            logger.info("Primary LLM initialized: %s", primary_name)
        except Exception as exc:
            logger.warning("Failed to initialize primary LLM: %s", exc)

    # 2. Initialize Fallback (Google Gemini)
    if settings.google_api_key and ChatGoogleGenerativeAI is not None:
        try:
            preferred_model = settings.google_model_name or "models/gemini-2.5-flash"
            if not preferred_model.startswith("models/"):
                preferred_model = f"models/{preferred_model}"

            fallback_llm = ChatGoogleGenerativeAI(
                model=preferred_model,
                google_api_key=settings.google_api_key,
                temperature=0.2,
                max_retries=1,
                timeout=45.0,
            )
            fallback_name = preferred_model
            logger.info("Fallback LLM initialized: %s", fallback_name)
        except Exception as exc:
            logger.warning("Failed to initialize fallback LLM: %s", exc)

    if not primary_llm and not fallback_llm:
        logger.error("No LLM could be initialized!")
        return None

    # If only one model available, it becomes both primary and sole option
    if not primary_llm:
        primary_llm = fallback_llm
        primary_name = fallback_name
        fallback_llm = None

    resilient = ResilientLLM(
        primary=primary_llm,
        fallback=fallback_llm,
        primary_model_name=primary_name,
        fallback_model_name=fallback_name,
        budget_threshold=0.85,
    )

    logger.info(
        "ResilientLLM ready: Primary=%s, Fallback=%s",
        primary_name, fallback_name if fallback_llm else "NONE",
    )
    return resilient
