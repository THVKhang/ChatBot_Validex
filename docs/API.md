# API Documentation — ChatBot Validex

> Complete REST API reference for the Validex AI Blog Generator.
> Base URL: `http://localhost:8000`
> Authentication: JWT Bearer Token (optional for chat, required for admin)

---

## Authentication

### Register
```
POST /api/auth/register
```
```json
// Request
{ "username": "alice", "password": "secret123" }

// Response 200
{ "message": "User created successfully", "user_id": 1 }

// Error 400
{ "detail": "Username already registered" }
{ "detail": "Password must be at least 6 characters long" }
```

### Login
```
POST /api/auth/login
Content-Type: application/x-www-form-urlencoded
```
```
username=alice&password=secret123
```
```json
// Response 200
{ "access_token": "eyJhbGci...", "token_type": "bearer" }

// Error 401
{ "detail": "Incorrect username or password" }
```

**Usage**: Include token in subsequent requests:
```
Authorization: Bearer eyJhbGci...
```

---

## Chat API

### Generate Blog (Synchronous)
```
POST /api/chat
```
```json
// Request
{
  "prompt": "Write a detailed guide about police checks in Australia",
  "session_id": null  // null for new session, UUID for follow-up
}

// Response 200
{
  "session_id": "a1b2c3d4-...",
  "parsed": {
    "intent": "create_blog",
    "topic": "police checks in Australia",
    "audience": "general audience",
    "tone": "clear_professional",
    "length": "medium",
    "language": "en"
  },
  "retrieved": [
    { "doc_id": "chunk_001", "score": 0.87, "snippet": "..." }
  ],
  "retrieval_meta": {
    "status": "sufficient",
    "confidence": 0.85,
    "top_score": 6,
    "docs_found": 5
  },
  "generated": {
    "title": "A Complete Guide to Police Checks in Australia",
    "outline": ["Introduction", "What You Need to Know", "..."],
    "draft": "# A Complete Guide...\n\n## Introduction\n...",
    "sources_used": ["acic.gov.au", "afp.gov.au"],
    "evaluation": {
      "relevance": 9, "coherence": 8, "factuality": 9,
      "overall": 8.7, "verdict": "ACCEPTED", "issues": []
    },
    "seo": { "score": 85, "issues": [], "keyword_density": 0.028 }
  },
  "runtime": {
    "quality_gate_blocked": false,
    "generation_mode": "multi-agent",
    "retrieval_mode": "hybrid",
    "external_knowledge_used": false,
    "token_budget": {
      "length_profile": "medium",
      "output_tokens_target": 1200,
      "recommended_top_k": 6
    }
  }
}
```

### Generate Blog (SSE Streaming)
```
POST /api/chat/stream
```
Same request body. Returns Server-Sent Events:
```
event: thinking
data: {"status": "Analyzing your request...", "step": "Parser"}

event: thinking
data: {"status": "Searching knowledge...", "step": "Researcher"}

event: chunk
data: {"chunk": "# A Complete Guide to "}

event: chunk
data: {"chunk": "Police Checks in Australia\n\n"}

event: done
data: { /* full ChatApiResponse */ }
```

### Upload Document
```
POST /api/chat/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>
```
Supports PDF and DOCX files (max 10MB).
```json
// Response 200
{ "filename": "report.pdf", "extracted_text": "..." }
```

### Export
```
POST /api/chat/export
```
```json
// Request
{ "markdown": "# My Blog\n\n## Section 1...", "format": "docx" }

// Response: Binary file download (Word/PDF/HTML)
```

### Session History
```
GET /api/chat/sessions?limit=50
```
```json
[
  {
    "session_id": "a1b2c3d4-...",
    "preview": "Write about police checks...",
    "created_at": "2026-07-27T10:00:00Z",
    "turn_count": 3
  }
]
```

```
GET /api/chat/sessions/{session_id}
```
```json
{
  "session_id": "a1b2c3d4-...",
  "turns": [
    {
      "user_prompt": "Write about police checks",
      "assistant_output": "# A Complete Guide...",
      "parsed_intent": "create_blog",
      "parsed_topic": "police checks"
    }
  ]
}
```

---

## Reports API

### CRUD Operations

```
POST   /api/reports           → Save report
GET    /api/reports           → List reports (?limit=50)
GET    /api/reports/{id}      → Get report detail
DELETE /api/reports/{id}      → Delete report
```

### Status Workflow
```
PATCH /api/reports/{id}/status
```
```json
{ "status": "Reviewed" }  // Valid: Draft → Reviewed → Approved
```

### Publish
```
POST /api/reports/{id}/publish?output_format=markdown
```
Only `Approved` reports can be published.

### Feedback
```
POST /api/reports/{id}/feedback
```
```json
{ "rating": 1, "comment": "Great article!" }  // rating: 1 (up) or -1 (down)
```

---

## Admin API

All admin endpoints require `Authorization: Bearer <admin_token>`.

### Knowledge Base Management
```
POST /api/admin/ingest          → Trigger re-ingestion (background job)
GET  /api/admin/ingest/status   → Check job status
POST /api/admin/discover        → AI Discovery Agent: find new data sources
GET  /api/admin/crawl-history   → Recent crawl job logs
```

### User Management
```
GET /api/admin/users            → List all users
```

### HITL (Human-in-the-Loop) Reviews
```
GET  /api/admin/pending-reviews    → List articles pending review
POST /api/admin/reviews/{run_id}   → Approve or reject
```
```json
{ "action": "Approve", "feedback": "Looks good" }
// or
{ "action": "Reject", "feedback": "Needs more sources" }
```

### Analytics Dashboard
```
GET /api/admin/analytics/tokens    → Token usage over 30 days
GET /api/admin/analytics/quality   → Editor verdict distribution
GET /api/admin/analytics/cache     → Semantic cache statistics
GET /api/admin/analytics/feedback  → User feedback statistics
GET /api/admin/token-usage         → Current + 7-day token usage
```

### Blog Scheduling
```
POST   /api/admin/schedule
GET    /api/admin/schedule
DELETE /api/admin/schedule/{id}
```
```json
// Create schedule
{
  "topic": "Police check processing times",
  "language": "en",
  "cron_expression": "0 9 * * MON",  // Every Monday 9 AM
  "is_active": true
}
```

---

## Monitoring

### Health Check
```
GET /api/health
```
Returns status of all system components: database, Redis, connection pool, sessions, LLM availability.

### Metrics
```
GET /api/metrics           → JSON format
GET /metrics               → Prometheus text format
```

### Token Budget
```
GET /api/user/token-budget
```
```json
{
  "tier": "free",
  "daily_limit": 25000,
  "used_today": 3200,
  "remaining": 21800,
  "reset_at": "2026-07-28T00:00:00Z"
}
```

---

## Error Codes

| HTTP Code | Meaning |
|-----------|---------|
| `400` | Bad request (invalid input, empty prompt) |
| `401` | Authentication required |
| `403` | Admin access required |
| `404` | Resource not found |
| `409` | Conflict (invalid status transition, job already running) |
| `429` | Rate limit exceeded |
| `503` | Server busy (LLM semaphore full) |
| `504` | Request timeout (>180s) |

---

## Rate Limits

| Limit | Default | Configurable |
|-------|---------|-------------|
| Requests per minute (per IP) | 30 | `RATE_LIMIT_PER_MINUTE` |
| Burst per 10 seconds (per IP) | 5 | `BURST_LIMIT_PER_10S` |
| Concurrent requests (per IP) | 3 | `MAX_CONCURRENT_PER_IP` |
| Global LLM concurrency | 10 | `MAX_CONCURRENT_LLM_REQUESTS` |
