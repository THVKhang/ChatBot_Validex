from collections import defaultdict
from collections import deque
import importlib
import time
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic import Field

from app.config import settings
from app.langchain_pipeline import pipeline
from app.main import process_prompt
from app.publisher import build_publish_output
from app.report_store import delete_report
from app.report_store import get_report
from app.report_store import list_reports
from app.report_store import save_report
from app.report_store import update_report_status
from app.session_manager import SessionManager
from app.source_analytics import fetch_knowledge_health
from app.source_analytics import fetch_source_analytics


class ChatRequest(BaseModel):
    prompt: str
    session_id: str | None = None


class BlogSectionPayload(BaseModel):
    heading: str
    body: str
    image_url: str
    image_alt: str


class GeneratedPayload(BaseModel):
    title: str
    outline: list[str] = Field(default_factory=list)
    draft: str
    sources_used: list[str] = Field(default_factory=list)
    sections: list[BlogSectionPayload] = Field(default_factory=list)


class SaveReportRequest(BaseModel):
    prompt: str
    generated: GeneratedPayload
    session_id: str | None = None


class UpdateReportStatusRequest(BaseModel):
    status: Literal["Draft", "Reviewed", "Approved"]


app = FastAPI(title="AI Blog Generator API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, SessionManager] = defaultdict(SessionManager)
_rate_limit_window: dict[str, deque[float]] = defaultdict(deque)
_metrics: dict[str, object] = {
    "chat_requests_total": 0,
    "chat_errors_total": 0,
    "quality_gate_blocked_total": 0,
    "generation_mode_count": defaultdict(int),
    "retrieval_mode_count": defaultdict(int),
    "latency_ms_samples": deque(maxlen=max(20, settings.metrics_window_size)),
}


def _redis_client():
    if not settings.use_redis_rate_limit:
        return None
    try:
        redis_module = importlib.import_module("redis")
    except Exception:
        return None
    redis_cls = getattr(redis_module, "Redis", None)
    if redis_cls is None:
        return None
    try:
        return redis_cls.from_url(settings.redis_url)
    except Exception:
        return None


_redis = _redis_client()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not settings.use_rate_limit:
        return await call_next(request)

    if request.url.path != "/api/chat":
        return await call_next(request)

    client_id = request.client.host if request.client else "unknown"
    if _redis is not None:
        key = f"rate_limit:{client_id}:{int(time.time() // 60)}"
        try:
            count = _redis.incr(key)
            if count == 1:
                _redis.expire(key, 61)
            if count > max(1, settings.rate_limit_per_minute):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please retry in a minute."},
                )
        except Exception:
            pass

    now = time.time()
    window = _rate_limit_window[client_id]
    while window and now - window[0] > 60:
        window.popleft()

    if len(window) >= max(1, settings.rate_limit_per_minute):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please retry in a minute."},
        )

    window.append(now)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    runtime = pipeline.runtime_status()
    return {
        "status": "ok",
        "runtime": runtime,
        "flags": {
            "use_live_llm": settings.use_live_llm,
            "use_pinecone_retrieval": settings.use_pinecone_retrieval,
            "use_agentic_rag": settings.use_agentic_rag,
            "use_rate_limit": settings.use_rate_limit,
            "use_redis_rate_limit": settings.use_redis_rate_limit,
        },
    }


@app.get("/api/metrics")
def metrics() -> dict:
    latency_samples = list(_metrics["latency_ms_samples"])
    avg_latency = round(sum(latency_samples) / len(latency_samples), 2) if latency_samples else 0.0
    p95_latency = 0.0
    if latency_samples:
        ordered = sorted(latency_samples)
        index = int(0.95 * (len(ordered) - 1))
        p95_latency = round(float(ordered[index]), 2)

    return {
        "chat_requests_total": _metrics["chat_requests_total"],
        "chat_errors_total": _metrics["chat_errors_total"],
        "quality_gate_blocked_total": _metrics["quality_gate_blocked_total"],
        "generation_mode_count": dict(_metrics["generation_mode_count"]),
        "retrieval_mode_count": dict(_metrics["retrieval_mode_count"]),
        "latency": {
            "samples": len(latency_samples),
            "avg_ms": avg_latency,
            "p95_ms": p95_latency,
        },
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    start = time.perf_counter()
    session_id = request.session_id or str(uuid4())
    session = _sessions[session_id]
    _metrics["chat_requests_total"] += 1

    try:
        payload = process_prompt(request.prompt, session)
    except Exception:
        _metrics["chat_errors_total"] += 1
        raise

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _metrics["latency_ms_samples"].append(elapsed_ms)

    runtime = payload.get("runtime", {}) if isinstance(payload, dict) else {}
    generation_mode = str(runtime.get("generation_mode", "unknown"))
    retrieval_mode = str(runtime.get("retrieval_mode", "unknown"))
    _metrics["generation_mode_count"][generation_mode] += 1
    _metrics["retrieval_mode_count"][retrieval_mode] += 1
    if runtime.get("quality_gate_blocked"):
        _metrics["quality_gate_blocked_total"] += 1

    return {"session_id": session_id, **payload}


@app.post("/api/reports")
def create_report(request: SaveReportRequest) -> dict:
    report = save_report(
        prompt=request.prompt,
        title=request.generated.title,
        outline=request.generated.outline,
        draft=request.generated.draft,
        sources_used=request.generated.sources_used,
        sections=[section.model_dump() for section in request.generated.sections],
        session_id=request.session_id,
    )
    return {"report": report}


@app.get("/api/reports")
def reports(limit: int = 50) -> dict:
    return {"reports": list_reports(limit=limit)}


@app.get("/api/reports/{report_id}")
def report_detail(report_id: str) -> dict:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"report": report}


@app.delete("/api/reports/{report_id}")
def report_delete(report_id: str) -> dict:
    deleted = delete_report(report_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"status": "deleted", "report_id": report_id}


@app.patch("/api/reports/{report_id}/status")
def report_status_update(report_id: str, request: UpdateReportStatusRequest) -> dict:
    report, error = update_report_status(report_id, request.status)
    if error == "not_found":
        raise HTTPException(status_code=404, detail="Report not found")
    if error == "invalid_transition":
        raise HTTPException(
            status_code=409,
            detail="Invalid status transition. Use Draft -> Reviewed -> Approved.",
        )
    if error == "invalid_status":
        raise HTTPException(status_code=400, detail="Invalid report status")
    return {"report": report}


@app.post("/api/reports/{report_id}/publish")
def report_publish(report_id: str, output_format: Literal["markdown", "html"] = "markdown") -> dict:
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    status = str(report.get("status", "Draft"))
    if status != "Approved":
        raise HTTPException(
            status_code=409,
            detail="Only Approved reports can be published.",
        )

    try:
        output = build_publish_output(report, output_format=output_format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "report_id": report_id,
        "status": "published",
        "report_status": status,
        "output": output,
    }


@app.get("/api/source-analytics")
def source_analytics() -> dict:
    try:
        return fetch_source_analytics()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/knowledge/health")
def knowledge_health() -> dict:
    try:
        return fetch_knowledge_health()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
