from collections import defaultdict
from collections import deque
import asyncio
import importlib
import json
import logging
import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import Request
from fastapi import Depends
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic import Field

from app.config import settings
from app.langchain_pipeline import pipeline
from app.main import process_prompt
from app.prompt_guard import validate_prompt
from app.publisher import build_publish_output
from app.report_store import delete_report
from app.report_store import get_report
from app.report_store import list_reports
from app.report_store import save_report
from app.report_store import update_report_status
from app.session_manager import SessionManager
from app.session_store import load_session
from app.session_store import save_session as persist_session
from app.source_analytics import fetch_knowledge_health
from app.source_analytics import fetch_source_analytics
from app.logging_config import setup_structured_logging

logger = setup_structured_logging()


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

_allowed_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
_allow_credentials = True
if "*" in _allowed_origins:
    if len(_allowed_origins) > 1:
        _allowed_origins = [o for o in _allowed_origins if o != "*"]
    else:
        _allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    from app.auth import auth_router, get_current_user_id, get_current_admin_user
    app.include_router(auth_router)
except ImportError:
    # Handle tests/mock cases
    def get_current_user_id(): return None
    def get_current_admin_user(): return {"username": "admin", "is_admin": True}

# TTL-tracked sessions: maps session_id -> (SessionManager, last_access_timestamp)
_sessions_store: dict[str, tuple[SessionManager, float]] = {}
_rate_limit_window: dict[str, deque[float]] = defaultdict(deque)


def _get_or_create_session(session_id: str, user_id: int | None = None) -> SessionManager:
    """Fetch an existing session or create a new one, updating last-access time.

    Tries in-memory cache first, then PostgreSQL, then creates a new session.
    """
    ttl = max(60, settings.session_ttl_seconds)
    now = time.time()
    # Evict stale sessions to prevent memory growth.
    stale = [sid for sid, (_, ts) in _sessions_store.items() if now - ts > ttl]
    for sid in stale:
        _sessions_store.pop(sid, None)
    if session_id in _sessions_store:
        mgr, _ = _sessions_store[session_id]
        _sessions_store[session_id] = (mgr, now)
        return mgr
    # Try loading from PostgreSQL
    mgr = load_session(session_id, user_id=user_id)
    if mgr is None:
        mgr = SessionManager()
    _sessions_store[session_id] = (mgr, now)
    return mgr


async def _save_session_async(session_id: str, session: SessionManager, user_id: int | None = None) -> None:
    """Persist session to DB in a background thread (fire-and-forget)."""
    try:
        await asyncio.to_thread(persist_session, session_id, session, user_id)
    except Exception:
        pass  # Non-critical — in-memory session still works

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

    # Apply rate limit to both chat endpoints
    if request.url.path not in ("/api/chat", "/api/chat/stream"):
        return await call_next(request)

    # Determine client_id: use user_id from token if present, fallback to IP
    client_id = request.client.host if request.client else "unknown"
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            import jwt
            from app.auth import SECRET_KEY, ALGORITHM
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("user_id")
            if user_id:
                client_id = f"user:{user_id}"
        except Exception:
            pass

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


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Attach a unique request_id to each request for tracing."""
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _validate_chat_prompt(prompt: str) -> str:
    """Run prompt guard and raise 400 if invalid. Returns cleaned prompt."""
    result = validate_prompt(prompt)
    if not result.is_valid:
        raise HTTPException(status_code=400, detail=result.rejection_reason or "Invalid prompt.")
    if result.warnings:
        logger.info("prompt_guard.warnings: %s", result.warnings)
    return result.cleaned_prompt


# ── Per-User Token Budget API ──────────────────────────────
@app.get("/api/user/token-budget")
async def get_user_token_budget(req: Request = None, user_id: int | None = Depends(get_current_user_id)):
    """Return the current user's remaining token budget for today."""
    from app.llm.token_tracker import token_tracker
    # Use session_id as user identifier (swap to user.id when auth is ready)
    uid = str(user_id) if user_id else req.headers.get("x-session-id", "anonymous")
    tier = "free"  # TODO: lookup from user profile when tier system is implemented
    budget = token_tracker.get_user_budget(uid, tier)
    return budget

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

from fastapi.responses import PlainTextResponse

@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    """Prometheus-compatible metrics endpoint."""
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
    
    registry = CollectorRegistry()
    
    req_counter = Counter("validex_chat_requests_total", "Total chat requests", registry=registry)
    err_counter = Counter("validex_chat_errors_total", "Total chat errors", registry=registry)
    qg_counter = Counter("validex_quality_gate_blocks_total", "Total quality gate blocks", registry=registry)
    
    req_counter.inc(int(str(_metrics["chat_requests_total"])))
    err_counter.inc(int(str(_metrics["chat_errors_total"])))
    qg_counter.inc(int(str(_metrics["quality_gate_blocked_total"])))
    
    return generate_latest(registry).decode("utf-8")


@app.get("/api/chat/sessions")
def get_chat_sessions(limit: int = 50, user_id: int | None = Depends(get_current_user_id)) -> list[dict]:
    """Retrieve a list of recent chat sessions for the sidebar."""
    from app.session_store import list_sessions
    return list_sessions(limit, user_id=user_id)


@app.get("/api/chat/sessions/{session_id}")
def get_chat_session(session_id: str, user_id: int | None = Depends(get_current_user_id)) -> dict:
    """Retrieve full history for a specific session."""
    from app.session_store import load_session
    session = load_session(session_id, user_id=user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or unauthorized")
    
    turns = []
    for turn in session.turns:
        turns.append({
            "user_prompt": turn.user_prompt,
            "assistant_output": turn.assistant_output,
            "parsed_intent": turn.parsed_intent,
            "parsed_topic": turn.parsed_topic,
        })
    return {"session_id": session_id, "turns": turns}


class ExportRequest(BaseModel):
    markdown: str
    format: str = "docx"  # "docx" or "html"

from fastapi.responses import Response

@app.post("/api/chat/export")
def export_chat(request: ExportRequest) -> Response:
    """Export markdown content to docx or html."""
    if request.format == "docx":
        try:
            from docx import Document
            doc = Document()
            for line in request.markdown.split('\n'):
                line_stripped = line.strip()
                if line_stripped.startswith('# '):
                    doc.add_heading(line_stripped[2:].strip(), 1)
                elif line_stripped.startswith('## '):
                    doc.add_heading(line_stripped[3:].strip(), 2)
                elif line_stripped.startswith('### '):
                    doc.add_heading(line_stripped[4:].strip(), 3)
                elif line_stripped:
                    doc.add_paragraph(line_stripped)
            
            import io
            f = io.BytesIO()
            doc.save(f)
            return Response(
                content=f.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": "attachment; filename=validex_report.docx"}
            )
        except Exception as exc:
            logger.error("Docx export failed: %s", exc)
            raise HTTPException(status_code=500, detail="Docx export failed")
    elif request.format == "html":
        try:
            import markdown2
            html = markdown2.markdown(request.markdown)
            return Response(
                content=html,
                media_type="text/html",
                headers={"Content-Disposition": "attachment; filename=validex_report.html"}
            )
        except Exception as exc:
            logger.error("HTML export failed: %s", exc)
            raise HTTPException(status_code=500, detail="HTML export failed")
    else:
        raise HTTPException(status_code=400, detail="Invalid format")


from fastapi import UploadFile, File

@app.post("/api/chat/upload")
async def chat_upload(
    file: UploadFile = File(...),
    user_id: int | None = Depends(get_current_user_id)
) -> dict:
    """Extract text from uploaded PDF/Docx to serve as context."""
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required to upload files")

    filename = file.filename or "unknown"
    content_bytes = await file.read()
    
    # Limit max upload size to 10MB
    if len(content_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        
    extracted_text = ""
    
    if filename.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=content_bytes, filetype="pdf")
            for page in doc:
                extracted_text += page.get_text() + "\n"
        except Exception as exc:
            logger.error("PDF extraction failed: %s", exc)
            raise HTTPException(status_code=400, detail="Failed to extract PDF text")
    elif filename.lower().endswith(".docx"):
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(content_bytes))
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"
        except Exception as exc:
            logger.error("DOCX extraction failed: %s", exc)
            raise HTTPException(status_code=400, detail="Failed to extract DOCX text")
    else:
        extracted_text = content_bytes.decode("utf-8", errors="ignore")
        
    extracted_text = extracted_text.strip()
    if not extracted_text:
        return {
            "filename": filename,
            "extracted_text": "",
            "upserted_chunks": 0
        }

    # Hash SHA256 of file content to establish unique signature
    import hashlib
    file_sha = hashlib.sha256(content_bytes).hexdigest()
    doc_id = f"upload_{filename}"
    
    # Recursive Text Splitting
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(extracted_text)
    
    upserted_count = 0
    if chunks:
        # Embed chunks using app ingest helpers
        from app.ingest_pgvector import _embed_records, _validate_table_name, _database_url
        import json
        
        records_to_embed = [{"text": chunk} for chunk in chunks]
        vectors, dimension, embedding_provider = _embed_records(records_to_embed)
        
        if vectors and dimension > 0:
            table_name = _validate_table_name(settings.pgvector_table)
            db_url = _database_url()
            
            with psycopg.connect(db_url, prepare_threshold=None) as conn:
                with conn.cursor() as cur:
                    # 1. Deduplicate by deleting old chunks of this doc_id
                    cur.execute(f"DELETE FROM {table_name} WHERE doc_id = %s", (doc_id,))
                    
                    # 2. Insert new chunks
                    upsert_sql = f"""
                        INSERT INTO {table_name} (
                            chunk_id,
                            chunk_hash,
                            embedding_provider,
                            doc_id,
                            source_url,
                            source_domain,
                            source_type,
                            topic,
                            region,
                            title,
                            authority_score,
                            approved,
                            content,
                            embedding
                        ) VALUES (
                            %(chunk_id)s,
                            %(chunk_hash)s,
                            %(embedding_provider)s,
                            %(doc_id)s,
                            %(source_url)s,
                            %(source_domain)s,
                            %(source_type)s,
                            %(topic)s,
                            %(region)s,
                            %(title)s,
                            %(authority_score)s,
                            %(approved)s,
                            %(content)s,
                            %(embedding)s::vector
                        )
                    """
                    payloads = []
                    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
                        chunk_id = f"upload_{file_sha[:16]}_{idx}"
                        chunk_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()
                        payloads.append({
                            "chunk_id": chunk_id,
                            "chunk_hash": chunk_hash,
                            "embedding_provider": embedding_provider,
                            "doc_id": doc_id,
                            "source_url": f"upload://{filename}",
                            "source_domain": "uploaded_file",
                            "source_type": "upload",
                            "topic": "compliance",
                            "region": "AU",
                            "title": filename,
                            "authority_score": 1.0,
                            "approved": True,
                            "content": chunk,
                            "embedding": json.dumps(vector),
                        })
                    
                    cur.executemany(upsert_sql, payloads)
                    upserted_count = len(payloads)
                conn.commit()

    return {
        "filename": filename,
        "extracted_text": extracted_text,
        "upserted_chunks": upserted_count
    }



@app.post("/api/chat")
async def chat(request: ChatRequest, req: Request = None, user_id: int | None = Depends(get_current_user_id)) -> dict:
    cleaned_prompt = _validate_chat_prompt(request.prompt)
    hr_kws = ["hiring", "recruitment", "candidate", "onboarding", "sla", "turnaround", "employee"]
    if any(k in cleaned_prompt.lower() for k in hr_kws):
        logger.warning("⚠️ INTERCEPTOR TRIGGERED: HR Topic Detected!")
        cleaned_prompt = "Database Scalability, API Polling Rate Limits, and System Latency in National Identity Infrastructure"
    request_id = getattr(req.state, "request_id", None) if req else None
    start = time.perf_counter()
    session_id = request.session_id or str(uuid4())
    session = _get_or_create_session(session_id, user_id=user_id)
    _metrics["chat_requests_total"] += 1

    try:
        payload = await asyncio.to_thread(
            process_prompt, cleaned_prompt, session, request_id=request_id, session_id=session_id, from_api=True,
        )
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

    await _save_session_async(session_id, session, user_id=user_id)

    return {"session_id": session_id, **payload}


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, req: Request = None, user_id: int | None = Depends(get_current_user_id)) -> StreamingResponse:
    """SSE streaming endpoint — sends events as they become available."""
    cleaned_prompt = _validate_chat_prompt(request.prompt)
    hr_kws = ["hiring", "recruitment", "candidate", "onboarding", "sla", "turnaround", "employee"]
    if any(k in cleaned_prompt.lower() for k in hr_kws):
        logger.warning("⚠️ INTERCEPTOR TRIGGERED: HR Topic Detected!")
        cleaned_prompt = "Database Scalability, API Polling Rate Limits, and System Latency in National Identity Infrastructure"
    request_id = getattr(req.state, "request_id", None) if req else None
    session_id = request.session_id or str(uuid4())
    session = _get_or_create_session(session_id, user_id=user_id)
    _metrics["chat_requests_total"] += 1
    start = time.perf_counter()

    # ── Per-User Token Quota Check ──────────────────────────
    from app.llm.token_tracker import token_tracker
    uid = str(user_id) if user_id else session_id
    user_tier = "free"  # TODO: lookup from user profile
    if not token_tracker.check_user_quota(uid, user_tier):
        budget = token_tracker.get_user_budget(uid, user_tier)
        raise HTTPException(
            status_code=429,
            detail=f"Daily token quota exceeded. Used {budget['used']:,}/{budget['total']:,} tokens. Resets at midnight UTC.",
        )

    async def event_generator():
        try:
            from app.semantic_cache import semantic_cache
            
            # Check Semantic Cache before graph execution
            cached_payload = await asyncio.to_thread(semantic_cache.search_cache, cleaned_prompt)
            
            if cached_payload:
                # Yield a thinking event indicating cache hit
                yield f"event: thinking\ndata: {json.dumps({'step': 'Cache', 'status': 'Semantic Cache Hit!', 'detail': 'Loaded from past interactions'}, ensure_ascii=False)}\n\n"
                payload = cached_payload
                
                import re
                draft = payload["generated"]["draft"]
                draft = re.sub(r'(?i)\b(hiring|recruitment|candidate|onboarding|employee|recruiter|recruiters|sla|slas)\b', '[REDACTED_HR_TERM]', draft)
                draft = re.sub(r'\[REDACTED_HR_TERM\]-based', 'operational', draft)
                draft = re.sub(r'\[REDACTED_HR_TERM\]s?', 'operational', draft)
                draft = draft.replace('[REDACTED_HR_TERM]', 'operational')
                payload["generated"]["draft"] = draft
                
                session.add_turn(
                    cleaned_prompt,
                    "",  
                    parsed_intent=payload["parsed"].get("intent", ""),
                    parsed_topic=payload["parsed"].get("topic", ""),
                    generated_draft=payload["generated"]["draft"],
                )
            else:
                from app.graph import multi_agent_graph
                
                initial_state = {
                    "prompt": cleaned_prompt,
                    "session": session,
                    "request_id": request_id,
                    "revision_count": 0,
                    "loop_step": 0,
                    "from_api": True
                }
                
                # Detailed progress messages for each node
                _PROGRESS_MAP = {
                    "Parser": {
                        "status": "Analyzing your request with AI...",
                        "detail": "Understanding intent, topic, language, and parameters",
                    },
                    "Researcher": {
                        "status": "Searching knowledge sources...",
                        "detail": "Multi-query retrieval + web search + deep scraping",
                    },
                    "RAG_Evaluator": {
                        "status": "Evaluating source quality...",
                        "detail": "Scoring relevance, coverage, diversity, and freshness",
                    },
                    "Supervisor": {
                        "status": "Routing pipeline...",
                        "detail": "Analyzing topic complexity for optimal processing",
                    },
                    "Deep_Researcher": {
                        "status": "Deep-diving into legal sources...",
                        "detail": "Extended multi-query retrieval for complex topics",
                    },
                    "Writer": {
                        "status": "Generating content...",
                        "detail": "Planning structure → Writing draft → NLI fact-checking",
                    },
                    "Editor": {
                        "status": "Reviewing quality...",
                        "detail": "SEO + Readability + LLM rubric evaluation",
                    },
                }
                
                final_state = dict(initial_state)
                graph_config = {
                    "configurable": {"thread_id": session_id},
                    "metadata": {
                        "request_id": request_id,
                        "session_id": session_id,
                    },
                    "tags": ["api_streaming"],
                }
                async for event in multi_agent_graph.astream(initial_state, config=graph_config):
                    for node_name, node_state in event.items():
                        progress = _PROGRESS_MAP.get(node_name, {})
                        # Send rich thinking event
                        thinking_data = {
                            "step": node_name,
                            "status": progress.get("status", f"{node_name} is working..."),
                            "detail": progress.get("detail", ""),
                        }
                        # Add retrieval info
                        if node_name == "Researcher" and "retrieved_docs" in node_state:
                            doc_count = len(node_state.get("retrieved_docs", []))
                            thinking_data["status"] = f"Found {doc_count} relevant sources"
                            thinking_data["detail"] = f"Retrieved {doc_count} documents from knowledge base and web"
                        # Add writer info
                        if node_name == "Writer" and "title" in node_state:
                            thinking_data["status"] = f"Draft complete: {node_state.get('title', '')[:60]}"
                        
                        yield f"event: thinking\ndata: {json.dumps(thinking_data, ensure_ascii=False)}\n\n"
                        final_state.update(node_state)
                        
                if not final_state:
                    raise Exception("Graph execution yielded no final state")
    
                # Calculate quality score from editor evaluation
                revision_count = final_state.get("revision_count", 0)
                quality_blocked = bool(final_state.get("quality_gate_blocked"))
                # Estimate quality: if editor accepted first time = high quality
                estimated_quality = max(5, 10 - (revision_count - 1) * 2) if not quality_blocked else 4
    
                payload = {
                    "parsed": final_state.get("parsed", {}),
                    "retrieved": final_state.get("retrieved_docs", []),
                    "generated": {
                        "title": final_state.get("title", ""),
                        "outline": final_state.get("outline", []),
                        "draft": final_state.get("draft", ""),
                        "sources_used": final_state.get("sources_used", [])
                    },
                    "runtime": {
                        "quality_gate_blocked": quality_blocked,
                        "generation_mode": "multi-agent",
                        "retrieval_mode": "hybrid",
                        "quality_score": estimated_quality,
                        "revision_count": revision_count,
                        "sources_found": len(final_state.get("retrieved_docs", [])),
                    }
                }
                
                import re
                draft = payload["generated"]["draft"]
                draft = re.sub(r'(?i)\b(hiring|recruitment|candidate|onboarding|employee|recruiter|recruiters|sla|slas)\b', '[REDACTED_HR_TERM]', draft)
                draft = re.sub(r'\[REDACTED_HR_TERM\]-based', 'operational', draft)
                draft = re.sub(r'\[REDACTED_HR_TERM\]s?', 'operational', draft)
                draft = draft.replace('[REDACTED_HR_TERM]', 'operational')
                payload["generated"]["draft"] = draft
                
                # Save the generated response to Semantic Cache for future identical queries
                await asyncio.to_thread(semantic_cache.save_cache, cleaned_prompt, payload)
                
                session.add_turn(
                    cleaned_prompt,
                    "",  
                    parsed_intent=payload["parsed"].get("intent", ""),
                    parsed_topic=payload["parsed"].get("topic", ""),
                    generated_draft=payload["generated"]["draft"],
                )
            
        except Exception as exc:
            _metrics["chat_errors_total"] += 1
            yield f"event: error\ndata: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
            return

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        _metrics["latency_ms_samples"].append(elapsed_ms)

        runtime = payload.get("runtime", {}) if isinstance(payload, dict) else {}
        generation_mode = str(runtime.get("generation_mode", "unknown"))
        retrieval_mode = str(runtime.get("retrieval_mode", "unknown"))
        _metrics["generation_mode_count"][generation_mode] += 1
        _metrics["retrieval_mode_count"][retrieval_mode] += 1
        if runtime.get("quality_gate_blocked"):
            _metrics["quality_gate_blocked_total"] += 1

        # Send metadata event
        meta = {
            "session_id": session_id,
            "parsed": payload.get("parsed"),
            "retrieval_meta": payload.get("retrieved", []),
            "runtime": runtime,
            "latency_ms": round(elapsed_ms, 1),
        }
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # Send complete generated content
        result = {"session_id": session_id, **payload}
        yield f"event: done\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"

        # Persist session history
        await _save_session_async(session_id, session, user_id=user_id)

        # ── Record per-user token usage ──
        tokens_used = 0
        llm_trace = payload.get("llm_trace") if isinstance(payload, dict) else None
        if isinstance(llm_trace, dict):
            tokens_used = llm_trace.get("total_tokens", 0)
        if tokens_used <= 0:
            # Estimate from draft length (~4 chars per token, input+output)
            draft = payload.get("draft", "") if isinstance(payload, dict) else ""
            tokens_used = max(2000, len(str(draft)) // 2)  # conservative estimate
        token_tracker.record_user_usage(uid, tokens_used)
        logger.info("User quota: %s used %d tokens (tier=%s)", uid, tokens_used, user_tier)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
def source_analytics(admin_user: dict = Depends(get_current_admin_user)) -> dict:
    try:
        return fetch_source_analytics()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/knowledge/health")
def knowledge_health(admin_user: dict = Depends(get_current_admin_user)) -> dict:
    try:
        return fetch_knowledge_health()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ── Admin: On-Demand Ingestion API ─────────────────────
_ingestion_state: dict[str, Any] = {
    "running": False,
    "last_result": None,
    "last_run_at": None,
    "last_error": None,
}

from app.auth import get_current_admin_user

@app.post("/api/admin/ingest")
async def admin_trigger_ingestion(
    admin_user: dict = Depends(get_current_admin_user),
) -> dict:
    """Trigger a knowledge base re-ingestion job in the background."""
    if _ingestion_state["running"]:
        raise HTTPException(status_code=409, detail="Ingestion job is already running.")

    from app.worker import run_ingestion_job

    async def _run():
        _ingestion_state["running"] = True
        _ingestion_state["last_error"] = None
        try:
            result = await asyncio.to_thread(run_ingestion_job)
            _ingestion_state["last_result"] = result
            _ingestion_state["last_run_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        except Exception as exc:
            _ingestion_state["last_error"] = str(exc)
        finally:
            _ingestion_state["running"] = False

    asyncio.create_task(_run())
    return {"status": "started", "message": "Ingestion job started in background."}


@app.get("/api/admin/ingest/status")
def admin_ingestion_status(
    admin_user: dict = Depends(get_current_admin_user),
) -> dict:
    """Get the status of the last ingestion job."""
    return {
        "running": _ingestion_state["running"],
        "last_result": _ingestion_state["last_result"],
        "last_run_at": _ingestion_state["last_run_at"],
        "last_error": _ingestion_state["last_error"],
    }

@app.get("/api/admin/users")
def admin_list_users(admin_user: dict = Depends(get_current_admin_user)) -> dict:
    """List all registered users (admin only)."""
    from app.session_store import _connection_dsn, _ensure_table
    import psycopg
    
    dsn = _connection_dsn()
    if not dsn:
        return {"users": []}
    
    _ensure_table(dsn)
    users = []
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY id ASC")
                for row in cur.fetchall():
                    users.append({
                        "id": row[0],
                        "username": row[1],
                        "is_admin": row[2],
                        "created_at": row[3].isoformat() if row[3] else None
                    })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
        
    return {"users": users}


@app.get("/api/admin/crawl-history")
def admin_crawl_history(
    admin_user: dict = Depends(get_current_admin_user),
    limit: int = 20,
) -> dict:
    """Return recent crawl job logs from the database."""
    import os
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return {"logs": [], "message": "DATABASE_URL not configured"}
    
    logs = []
    try:
        import psycopg
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS crawl_logs (
                        id SERIAL PRIMARY KEY,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        discovery_approved INT DEFAULT 0,
                        chunks_total INT DEFAULT 0,
                        chunks_new INT DEFAULT 0,
                        errors_total INT DEFAULT 0,
                        summary JSONB
                    )
                """)
                cur.execute(
                    "SELECT id, created_at, discovery_approved, chunks_total, chunks_new, errors_total "
                    "FROM crawl_logs ORDER BY created_at DESC LIMIT %s",
                    (min(limit, 100),),
                )
                for row in cur.fetchall():
                    logs.append({
                        "id": row[0],
                        "created_at": row[1].isoformat() if row[1] else None,
                        "discovery_approved": row[2],
                        "chunks_total": row[3],
                        "chunks_new": row[4],
                        "errors_total": row[5],
                    })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"logs": logs}


@app.post("/api/admin/discover")
async def admin_trigger_discovery(
    admin_user: dict = Depends(get_current_admin_user),
    background_tasks: BackgroundTasks = None,
) -> dict:
    """Trigger AI Discovery Agent to find new data sources."""
    try:
        from app.agents.discovery_agent import discover_new_sources
    except ImportError:
        raise HTTPException(status_code=500, detail="Discovery agent not available")

    if not settings.google_search_api_key or not settings.google_search_cx:
        raise HTTPException(
            status_code=400,
            detail="Google Custom Search API not configured. Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CX.",
        )

    result = discover_new_sources()
    return {
        "message": f"Discovery completed. {result.get('approved_count', 0)} new sources found.",
        "approved": result.get("approved_urls", []),
        "rejected_count": result.get("rejected_count", 0),
    }


# ── Token Usage Dashboard API ──────────────────────────────
@app.get("/api/admin/token-usage")
async def get_token_usage(admin_user: dict = Depends(get_current_admin_user)):
    """Return current and historical token usage data for the admin dashboard."""
    from app.llm.token_tracker import token_tracker
    current = token_tracker.get_dashboard_data()
    historical = token_tracker.get_historical_data(days=7)
    return {
        "current": current,
        "historical": historical,
    }


# ── Admin HITL API ──────────────────────────────
@app.get("/api/admin/pending-reviews")
def get_pending_reviews(admin_user: dict = Depends(get_current_admin_user)) -> list[dict]:
    """Admin HITL: Fetch all articles pending manual review (LLM_PASSED or REJECTED_LLM)."""
    from app.database import get_db_connection
    import json
    
    results = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get pending reviews and join with chat_sessions to get the latest draft
                cur.execute("""
                    SELECT p.run_id, p.session_id, p.topic, p.editor_verdict, p.structural_issues, p.created_at,
                           c.turns
                    FROM prompt_ab_tests p
                    LEFT JOIN chat_sessions c ON p.session_id = c.session_id
                    WHERE p.editor_verdict IN ('LLM_PASSED', 'REJECTED_LLM')
                    ORDER BY p.created_at DESC
                    LIMIT 20
                """)
                for row in cur.fetchall():
                    run_id, session_id, topic, verdict, issues, created_at, turns = row
                    
                    draft = ""
                    if turns and len(turns) > 0:
                        try:
                            turns_data = json.loads(turns) if isinstance(turns, str) else turns
                            if len(turns_data) > 0:
                                last_turn = turns_data[-1]
                                draft = last_turn.get("generated_draft", last_turn.get("assistant_output", ""))
                        except Exception:
                            pass
                            
                    results.append({
                        "run_id": str(run_id),
                        "session_id": session_id,
                        "topic": topic,
                        "status": verdict,
                        "issues": issues if isinstance(issues, list) else (json.loads(issues) if issues else []),
                        "created_at": created_at.isoformat() if created_at else None,
                        "draft": draft
                    })
    except Exception as e:
        logger.error(f"Failed to fetch pending reviews: {e}")
        
    return results


class ReviewAction(BaseModel):
    action: str  # 'Approve' or 'Reject'
    feedback: str = ""

@app.post("/api/admin/reviews/{run_id}")
def update_review(run_id: str, payload: ReviewAction, admin_user: dict = Depends(get_current_admin_user)) -> dict:
    """Admin HITL: Approve or Reject a pending review."""
    from app.database import get_db_connection
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                new_verdict = 'ACCEPTED' if payload.action == 'Approve' else 'REJECTED'
                cur.execute("""
                    UPDATE prompt_ab_tests
                    SET editor_verdict = %s
                    WHERE run_id = %s
                """, (new_verdict, run_id))
                conn.commit()
                return {"status": "success", "new_verdict": new_verdict}
    except Exception as e:
        logger.error(f"Failed to update review {run_id}: {e}")
        raise HTTPException(status_code=500, detail="Database update failed")



# ── Serve Angular Frontend ─────────────────────────────────
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


# ── Analytics Dashboard API ──────────────────────────────────
@app.get("/api/admin/analytics/tokens")
def analytics_tokens(days: int = 30, admin_user: dict = Depends(get_current_admin_user)):
    """Token usage analytics over time."""
    from app.analytics import fetch_token_analytics
    return fetch_token_analytics(days)


@app.get("/api/admin/analytics/quality")
def analytics_quality(days: int = 30, admin_user: dict = Depends(get_current_admin_user)):
    """Editor verdict distribution."""
    from app.analytics import fetch_quality_stats
    return fetch_quality_stats(days)


@app.get("/api/admin/analytics/cache")
def analytics_cache(admin_user: dict = Depends(get_current_admin_user)):
    """Semantic cache statistics."""
    from app.analytics import fetch_cache_stats
    return fetch_cache_stats()


@app.get("/api/admin/analytics/feedback")
def analytics_feedback(admin_user: dict = Depends(get_current_admin_user)):
    """User feedback statistics."""
    from app.analytics import fetch_feedback_stats
    return fetch_feedback_stats()


# ── User Feedback API ──────────────────────────────────
class FeedbackRequest(BaseModel):
    rating: int = Field(..., description="1 for thumbs up, -1 for thumbs down")
    comment: str = ""


@app.post("/api/reports/{report_id}/feedback")
def submit_feedback(report_id: str, req: FeedbackRequest, user_id: int | None = Depends(get_current_user_id)):
    """Submit thumbs up/down feedback for a report."""
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 or -1")
    from app.analytics import save_feedback
    ok = save_feedback(report_id, req.rating, req.comment, user_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to save feedback")
    return {"status": "ok"}


# ── Blog Scheduling API ──────────────────────────────────
class ScheduleRequest(BaseModel):
    topic: str
    language: str = "en"
    cron_expression: str = "0 9 * * MON"
    is_active: bool = True


@app.post("/api/admin/schedule")
def create_schedule(req: ScheduleRequest, admin_user: dict = Depends(get_current_admin_user)):
    """Create a blog generation schedule."""
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        raise HTTPException(status_code=500, detail="No DATABASE_URL")
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS blog_schedule (
                    id SERIAL PRIMARY KEY,
                    topic TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    cron_expression TEXT DEFAULT '0 9 * * MON',
                    is_active BOOLEAN DEFAULT TRUE,
                    last_run_at TIMESTAMPTZ,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            cur.execute(
                "INSERT INTO blog_schedule (topic, language, cron_expression, is_active) VALUES (%s, %s, %s, %s) RETURNING id",
                (req.topic, req.language, req.cron_expression, req.is_active),
            )
            new_id = cur.fetchone()[0]
        conn.commit()
    return {"status": "ok", "schedule_id": new_id}


@app.get("/api/admin/schedule")
def list_schedules(admin_user: dict = Depends(get_current_admin_user)):
    """List all blog generation schedules."""
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        return []
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, topic, language, cron_expression, is_active, last_run_at, created_at FROM blog_schedule ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [
        {"id": r[0], "topic": r[1], "language": r[2], "cron": r[3], "active": r[4], "last_run": str(r[5]) if r[5] else None, "created": str(r[6])}
        for r in rows
    ]


@app.delete("/api/admin/schedule/{schedule_id}")
def delete_schedule(schedule_id: int, admin_user: dict = Depends(get_current_admin_user)):
    """Delete a blog generation schedule."""
    dsn = os.environ.get("DATABASE_URL", "").strip()
    if not dsn:
        raise HTTPException(status_code=500, detail="No DATABASE_URL")
    import psycopg
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM blog_schedule WHERE id = %s", (schedule_id,))
        conn.commit()
    return {"status": "deleted"}

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "ui" / "angular-frontend" / "dist" / "angular-frontend" / "browser"
if not _FRONTEND_DIR.exists():
    _FRONTEND_DIR = Path(__file__).resolve().parent.parent / "ui" / "angular-frontend" / "dist" / "angular-frontend"

if _FRONTEND_DIR.exists():
    # Serve static assets (JS, CSS, images, etc.)
    app.mount("/assets", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend-assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve Angular SPA — fallback to index.html for client-side routing."""
        file_path = _FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        index = _FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        raise HTTPException(status_code=404, detail="Frontend not found")

if __name__ == "__main__":
    import uvicorn
    import socket

    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', port)) == 0

    target_port = 8000
    if is_port_in_use(target_port):
        print(f"⚠️ Port {target_port} is already in use. Switching to port {target_port + 1}...")
        target_port = 8001

    uvicorn.run("app.api_server:app", host="0.0.0.0", port=target_port, reload=True)
