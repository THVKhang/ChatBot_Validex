from collections import defaultdict
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.main import process_prompt
from app.session_manager import SessionManager


class ChatRequest(BaseModel):
    prompt: str
    session_id: str | None = None


app = FastAPI(title="AI Blog Generator API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_sessions: dict[str, SessionManager] = defaultdict(SessionManager)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid4())
    session = _sessions[session_id]
    payload = process_prompt(request.prompt, session)
    return {"session_id": session_id, **payload}
