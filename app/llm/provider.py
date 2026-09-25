"""Resilient LLM Provider — wraps LangChain LLMs with smart retry, budget-aware
routing, and automatic fallback.

When the primary model (e.g., Groq Llama 3.3 70B) hits a rate limit or exhausts
its daily token budget, the wrapper transparently routes the request to the
fallback model (e.g., Gemini Flash) without any caller-side changes.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Any

from app.llm.token_tracker import token_tracker

logger = logging.getLogger(__name__)

# Models that returned a *permanent* error (retired, misspelled, or not granted
# to this key). Retrying those is guaranteed waste, and so is re-discovering
# them on every later call: one dead model used to cost 3 requests plus 3s of
# backoff sleep on every single LLM invocation in the pipeline.
_DEAD_MODELS: set[str] = set()
_DEAD_MODELS_LOCK = threading.Lock()

# Substrings that identify an error no retry can fix.
_PERMANENT_ERROR_MARKERS = (
    "model_not_found",
    "does not exist",
    "no longer available",
    "is not found",
    "not_found",
    "invalid_api_key",
    "invalid api key",
    "unauthorized",
    "permission_denied",
    "error code: 401",
    "error code: 403",
)


# Backoff for throttling (429) and transient overload (503), per retry attempt.
_THROTTLE_BACKOFF = (4.0, 12.0)

_RETRY_AFTER_RE = re.compile(
    r"retry[-_ ]?after[\"'\s:=]+(\d+(?:\.\d+)?)|retryDelay[\"'\s:=]+(\d+(?:\.\d+)?)s",
    re.IGNORECASE,
)

# Minimum spacing between outbound LLM calls, to stay under provider
# requests-per-minute limits instead of bursting into them. The generation
# pipeline issues its calls back-to-back; without pacing it throttles itself.
_MIN_CALL_INTERVAL = float(os.getenv("LLM_MIN_CALL_INTERVAL", "0.5"))
_last_call_at = 0.0
_pace_lock = threading.Lock()


def _pace() -> None:
    """Block just long enough to keep calls _MIN_CALL_INTERVAL apart."""
    global _last_call_at
    if _MIN_CALL_INTERVAL <= 0:
        return
    with _pace_lock:
        wait = _MIN_CALL_INTERVAL - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def _retry_after_seconds(error: Exception, default: float) -> float:
    """Honour a provider-supplied Retry-After / retryDelay when present."""
    match = _RETRY_AFTER_RE.search(str(error))
    if match:
        value = match.group(1) or match.group(2)
        try:
            return max(0.5, min(60.0, float(value)))
        except (TypeError, ValueError):
            pass
    return default


# A per-day quota is not something a retry can outwait: it resets at midnight
# UTC, not in 12 seconds. Waiting on it turned one article into a 33-minute
# grind. Park the model instead and let the caller fail fast.
_QUOTA_EXHAUSTED: dict[str, float] = {}
_QUOTA_LOCK = threading.Lock()
_QUOTA_COOLDOWN = float(os.getenv("LLM_DAILY_QUOTA_COOLDOWN", "900"))

_DAILY_QUOTA_MARKERS = (
    "perday",
    "per day",
    "requestsperdaypermodel",
    "generaterequestsperdayperprojectpermodel",
    "free_tier_requests",
)


def _is_daily_quota_exhausted(error: Exception) -> bool:
    text = str(error).lower()
    if "429" not in text and "resource_exhausted" not in text:
        return False
    return any(marker in text.replace("-", "").replace("_", "") or marker in text
               for marker in _DAILY_QUOTA_MARKERS)


def _mark_quota_exhausted(model_name: str) -> None:
    with _QUOTA_LOCK:
        first = model_name not in _QUOTA_EXHAUSTED
        _QUOTA_EXHAUSTED[model_name] = time.monotonic() + _QUOTA_COOLDOWN
    if first:
        logger.error(
            "Daily quota exhausted for '%s'. Skipping it for %.0fs instead of retrying — "
            "a per-day limit does not clear on backoff.",
            model_name, _QUOTA_COOLDOWN,
        )


def is_quota_exhausted(model_name: str) -> bool:
    with _QUOTA_LOCK:
        until = _QUOTA_EXHAUSTED.get(model_name)
        if until is None:
            return False
        if time.monotonic() >= until:
            del _QUOTA_EXHAUSTED[model_name]
            return False
        return True


def reset_quota_state() -> None:
    """Clear quota cooldowns (used by tests)."""
    with _QUOTA_LOCK:
        _QUOTA_EXHAUSTED.clear()


def extract_text(response: Any) -> str:
    """Get the plain text out of an LLM response, whatever shape it arrives in.

    Older chat models set ``.content`` to a string. Gemini 3.x (and Anthropic
    models) return a LIST of content blocks instead, so ``str(response.content)``
    yields the Python repr — ``[{'type': 'text', 'text': '# Title...'}]`` — and
    that repr was being published as the article body.
    """
    if response is None:
        return ""

    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # {"type": "text", "text": ...} is the common shape; skip
                # non-text blocks (thinking, tool_use, images).
                if block.get("type") in (None, "text") and isinstance(block.get("text"), str):
                    parts.append(block["text"])
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content) if content is not None else ""


def _is_permanent_error(error: Exception) -> bool:
    """True when an error will recur identically no matter how often we retry."""
    text = str(error).lower()
    if "429" in text or "rate_limit" in text or "quota" in text:
        return False  # transient: throttling, not a broken model
    return any(marker in text for marker in _PERMANENT_ERROR_MARKERS)


def _mark_dead(model_name: str, error: Exception) -> None:
    with _DEAD_MODELS_LOCK:
        if model_name in _DEAD_MODELS:
            return
        _DEAD_MODELS.add(model_name)
    logger.error(
        "Model '%s' is unusable and will be skipped for the rest of this process: %s",
        model_name, str(error)[:200],
    )


def is_model_dead(model_name: str) -> bool:
    with _DEAD_MODELS_LOCK:
        return model_name in _DEAD_MODELS


def reset_dead_models() -> None:
    """Clear the dead-model cache (used by tests and after a config change)."""
    with _DEAD_MODELS_LOCK:
        _DEAD_MODELS.clear()


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

        # 2. Skip a primary already known to be unusable — no request, no sleep.
        last_error = None
        primary_unusable = (
            is_model_dead(self.primary_model_name)
            or is_quota_exhausted(self.primary_model_name)
        )
        if primary_unusable and self.fallback:
            logger.debug(
                "Skipping unusable primary %s, going straight to %s",
                self.primary_model_name, self.fallback_model_name,
            )
        else:
            # 3. Try primary with exponential backoff
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

                    # Permanent error (retired model, bad key) → retrying cannot
                    # help. Remember it and fall through to the fallback now.
                    if _is_permanent_error(exc):
                        _mark_dead(self.primary_model_name, exc)
                        break

                    # A per-day quota will not clear during this request. Park
                    # the model and move on instead of sleeping through it.
                    if _is_daily_quota_exhausted(exc):
                        _mark_quota_exhausted(self.primary_model_name)
                        break

                    # Throttling and transient overload are worth waiting out on
                    # the SAME model. Jumping straight to the fallback turned a
                    # recoverable 429 into a hard failure whenever the fallback
                    # was unavailable — which is exactly how sections ended up
                    # half-written in the published draft.
                    is_overloaded = "503" in error_str or "unavailable" in error_str or "overloaded" in error_str
                    if attempt < 2:
                        if is_rate_limit or is_overloaded:
                            sleep_time = _retry_after_seconds(exc, default=_THROTTLE_BACKOFF[attempt])
                            logger.info("Throttled by %s, waiting %.1fs before retry", self.primary_model_name, sleep_time)
                        else:
                            sleep_time = 2 ** attempt
                            logger.info("Retrying in %ds...", sleep_time)
                        time.sleep(sleep_time)

        # 4. Primary unavailable → fallback
        if self.fallback:
            if last_error is not None:
                logger.warning("Primary unavailable, falling back to %s", self.fallback_model_name)
            try:
                return self._invoke_with_tracking(
                    self.fallback, self.fallback_model_name, input_data, estimated_input, **kwargs
                )
            except Exception as fb_exc:
                if _is_permanent_error(fb_exc):
                    _mark_dead(self.fallback_model_name, fb_exc)
                elif _is_daily_quota_exhausted(fb_exc):
                    _mark_quota_exhausted(self.fallback_model_name)
                logger.error("Last-resort fallback also failed: %s", fb_exc)
                raise fb_exc from last_error

        raise last_error  # type: ignore[misc]

    def with_structured_output(self, schema: Any, **kwargs: Any) -> "ResilientLLM":
        """Bind ``schema`` to both legs and keep the resilience chain intact.

        The pipeline asks the LLM for a schema-validated blog via
        ``with_structured_output``. ResilientLLM used not to expose the method
        at all, so ``hasattr(self._llm, "with_structured_output")`` was False and
        structured generation silently returned None on every call no matter what
        USE_STRUCTURED_OUTPUT said — the flag looked like a toggle but was wired
        to nothing.

        Binding each leg separately (instead of returning the bare primary's
        bound model) is what preserves retry, budget routing and fallback for
        structured calls: a primary outage would otherwise kill structured
        generation outright rather than handing over to the fallback.
        """
        if not hasattr(self.primary, "with_structured_output"):
            raise AttributeError(
                f"Primary model {self.primary_model_name} does not support structured output"
            )

        bound_primary = self.primary.with_structured_output(schema, **kwargs)

        bound_fallback = None
        if self.fallback is not None and hasattr(self.fallback, "with_structured_output"):
            try:
                bound_fallback = self.fallback.with_structured_output(schema, **kwargs)
            except Exception as exc:  # a provider that cannot bind this schema
                logger.warning(
                    "Fallback %s cannot bind structured schema, structured calls "
                    "will run without a fallback leg: %s",
                    self.fallback_model_name, exc,
                )

        return ResilientLLM(
            primary=bound_primary,
            fallback=bound_fallback,
            primary_model_name=self.primary_model_name,
            fallback_model_name=self.fallback_model_name,
            budget_threshold=self.budget_threshold,
        )

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
        _pace()
        try:
            result = llm.invoke(input_data, **kwargs)

            # Estimate output tokens
            output_text = extract_text(result)
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
        max_tokens=settings.writer_max_output_tokens,
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
