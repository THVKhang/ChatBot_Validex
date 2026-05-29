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


# ── Helper to build OpenAI + Gemini pair ──────────────────────────────
def _build_openai_gemini_pair(
    settings: Any,
    openai_model: str,
    google_model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[Any, str, Any, str]:
    """Build primary (OpenAI/Groq) + fallback (Gemini) LLM pair.
    Returns (primary_llm, primary_name, fallback_llm, fallback_name).
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        ChatOpenAI = None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        ChatGoogleGenerativeAI = None

    openai_llm = None
    openai_name = "unknown"
    google_llm = None
    google_name = "unknown"

    if settings.openai_api_key and ChatOpenAI is not None:
        try:
            kwargs = {
                "model": openai_model,
                "api_key": settings.openai_api_key,
                "temperature": temperature,
                "max_retries": 0,
                "timeout": 30.0,
                "max_tokens": max_tokens,
            }
            if hasattr(settings, "openai_base_url") and settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            openai_llm = ChatOpenAI(**kwargs)
            openai_name = openai_model
        except Exception as exc:
            logger.warning("Failed to init OpenAI %s: %s", openai_model, exc)

    if settings.google_api_key and ChatGoogleGenerativeAI is not None:
        try:
            gmodel = google_model if google_model.startswith("models/") else f"models/{google_model}"
            google_llm = ChatGoogleGenerativeAI(
                model=gmodel,
                google_api_key=settings.google_api_key,
                temperature=temperature,
                max_retries=1,
                timeout=45.0,
                max_output_tokens=max_tokens,
            )
            google_name = gmodel
        except Exception as exc:
            logger.warning("Failed to init Google %s: %s", google_model, exc)

    # Swap primary and fallback if Google provider is selected
    provider = getattr(settings, "llm_provider", "auto").lower()
    if provider == "google":
        primary_llm, primary_name = google_llm, google_name
        fallback_llm, fallback_name = openai_llm, openai_name
    else:
        primary_llm, primary_name = openai_llm, openai_name
        fallback_llm, fallback_name = google_llm, google_name

    return primary_llm, primary_name, fallback_llm, fallback_name


def build_resilient_llm(settings: Any) -> ResilientLLM | None:
    """Factory: builds the MAIN ResilientLLM (used by Writer for creative generation).

    Temperature is set to writer_temperature (default 0.7 for creative prose).
    """
    if not settings.use_live_llm:
        return None

    writer_temp = getattr(settings, "writer_temperature", 0.7)
    primary_llm, primary_name, fallback_llm, fallback_name = _build_openai_gemini_pair(
        settings,
        openai_model=settings.model_name,
        google_model=settings.google_model_name or "models/gemini-2.5-flash",
        temperature=writer_temp,
        max_tokens=2048,
    )

    if not primary_llm and not fallback_llm:
        logger.error("No LLM could be initialized!")
        return None

    if not primary_llm:
        primary_llm, primary_name = fallback_llm, fallback_name
        fallback_llm = None

    resilient = ResilientLLM(
        primary=primary_llm, fallback=fallback_llm,
        primary_model_name=primary_name, fallback_model_name=fallback_name,
        budget_threshold=0.85,
    )
    logger.info(
        "ResilientLLM ready: Primary=%s (temp=%.1f), Fallback=%s",
        primary_name, writer_temp, fallback_name if fallback_llm else "NONE",
    )
    return resilient


def build_resilient_fast_llm(settings: Any) -> ResilientLLM | None:
    """Factory: builds a FAST ResilientLLM for planning/review (low temperature)."""
    if not settings.use_live_llm:
        return None

    fast_model = getattr(settings, "fast_model_name", settings.model_name)
    fast_google = getattr(settings, "google_fast_model_name", settings.google_model_name or "models/gemini-2.5-flash")

    primary_llm, primary_name, fallback_llm, fallback_name = _build_openai_gemini_pair(
        settings,
        openai_model=fast_model,
        google_model=fast_google,
        temperature=0.2,
        max_tokens=2048,
    )

    if not primary_llm and not fallback_llm:
        logger.error("No Fast LLM could be initialized!")
        return None

    if not primary_llm:
        primary_llm, primary_name = fallback_llm, fallback_name
        fallback_llm = None

    resilient = ResilientLLM(
        primary=primary_llm, fallback=fallback_llm,
        primary_model_name=primary_name, fallback_model_name=fallback_name,
        budget_threshold=0.85,
    )
    logger.info("Fast ResilientLLM ready: Primary=%s, Fallback=%s", primary_name, fallback_name if fallback_llm else "NONE")
    return resilient


def build_editor_llm(settings: Any) -> ResilientLLM | None:
    """Factory: builds a SEPARATE LLM for the Editor node — breaks the Debate Agent Problem.

    Strategy:
        1. If EDITOR_MODEL_NAME is set → use a completely different model (best fix)
        2. Otherwise → use same model but with editor_temperature (0.1 strict) = divergence mode

    This ensures the Editor has a genuinely different perspective from the Writer,
    preventing the self-validation bias inherent in same-model Writer↔Editor loops.
    """
    if not settings.use_live_llm:
        return None

    editor_temp = getattr(settings, "editor_temperature", 0.1)
    editor_model = getattr(settings, "editor_model_name", "").strip()
    editor_google = getattr(settings, "editor_google_model_name", "").strip()

    # Determine which models to use for editor
    openai_model = editor_model or settings.model_name
    google_model = editor_google or settings.google_model_name or "models/gemini-2.5-flash"

    primary_llm, primary_name, fallback_llm, fallback_name = _build_openai_gemini_pair(
        settings,
        openai_model=openai_model,
        google_model=google_model,
        temperature=editor_temp,
        max_tokens=1024,  # Editor needs less output than Writer
    )

    if not primary_llm and not fallback_llm:
        logger.warning("No Editor LLM could be initialized, will fall back to primary")
        return None

    if not primary_llm:
        primary_llm, primary_name = fallback_llm, fallback_name
        fallback_llm = None

    # Tag the name so logs clearly show divergence
    if not editor_model and not editor_google:
        primary_name = f"{primary_name}@temp{editor_temp}"

    resilient = ResilientLLM(
        primary=primary_llm, fallback=fallback_llm,
        primary_model_name=primary_name, fallback_model_name=fallback_name,
        budget_threshold=0.90,
    )

    mode = "mixed-model" if editor_model or editor_google else "divergence"
    logger.info("Editor ResilientLLM ready: %s mode=%s (Debate Agent fix active)", primary_name, mode)
    return resilient
