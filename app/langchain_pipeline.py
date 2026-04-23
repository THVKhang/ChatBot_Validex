from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import replace
import hashlib
import json
import importlib
import logging
import math
import os
import random
import re
from urllib.parse import quote_plus
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen
from typing import Any

import psycopg
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from app.cache import response_cache
from app.config import settings
from app.generator import build_section_image_url
from app.generator import build_sections
from app.generator import extract_requested_image_limit
from app.generator import format_title
from app.generator import GeneratedBlog
from app.generator import generate_blog_output
from app.generator import render_markdown_blog
from app.parser import ParsedPrompt
from app.parser import parse_user_input
from app.retriever import RetrievalDecision
from app.retriever import RetrievedDoc
from app.retriever import retrieve_with_guard

try:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    ChatOpenAI = None
    OpenAIEmbeddings = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover - optional dependency
    ChatGoogleGenerativeAI = None
    GoogleGenerativeAIEmbeddings = None


logger = logging.getLogger(__name__)

MISSING_INTERNAL_DATA_TEXT = "Dữ liệu nội bộ hiện tại không đề cập đến vấn đề này."
SOURCE_LINE_PREFIX = "Nguồn:"
SOURCES_SECTION_HEADING = "## Danh mục nguồn tham khảo"
MAX_LLM_FAILURE_RECORDS = 8

@dataclass
class RetrievalBundle:
    decision: RetrievalDecision
    documents: list[Document]


@dataclass
class TokenBudgetPlan:
    length_profile: str
    output_tokens: int
    input_tokens_min: int
    input_tokens_target: int
    input_tokens_max: int
    recommended_top_k: int


class BlogSectionSchema(BaseModel):
    header: str = Field(min_length=3, max_length=120)
    content: str = Field(min_length=40)
    image_search_keyword: str = Field(min_length=2, max_length=80)


class BlogStructuredOutput(BaseModel):
    title: str = Field(min_length=8, max_length=140)
    introduction: str = Field(min_length=80)
    sections: list[BlogSectionSchema] = Field(min_length=3, max_length=10)
    conclusion: str = Field(min_length=80)
    meta_tags: str = Field(default="")


class PromptParseSchema(BaseModel):
    intent: str = Field(default="create_blog")
    topic: str = Field(default="current draft")
    tone: str = Field(default="clear_professional")
    audience: str = Field(default="general audience")
    length: str = Field(default="medium")
    custom_instructions: str = Field(default="")


class LangChainRAGPipeline:
    """LangChain-based orchestration for parse -> retrieve -> generate."""

    def __init__(self) -> None:
        self._llm = self._build_llm()
        self._embedding_model = self._build_embedding_model()
        self._vector_store = self._build_vector_store()
        self._agent_executor = self._build_agent_executor()
        self._last_generation_mode = "fallback"
        # Circuit breaker state
        self._cb_consecutive_failures = 0
        self._cb_open_until: float = 0.0  # timestamp until which the breaker stays open
        self._parse_chain = RunnableLambda(self._parse)
        self._retrieve_chain = RunnableLambda(self._retrieve)
        system_prompt = """
            Bạn là "Chuyên gia Pháp lý và Nội dung" tại Validex (Australia). Nhiệm vụ của bạn là soạn thảo các bài blog chuyên nghiệp,
            chính xác về background checks, compliance và an toàn tại nơi làm việc theo tiêu chuẩn Úc.

            ### 🛡 NGUYÊN TẮC CỐT LÕI (GROUNDING RULES):
            1. CHỈ sử dụng thông tin từ dữ liệu nội bộ được cung cấp trong context.
            2. Tuyệt đối KHÔNG tự ý suy diễn hoặc dùng kiến thức bên ngoài nếu dữ liệu nội bộ không đề cập.
            3. Nếu context không có đủ thông tin để trả lời yêu cầu, phải ghi chính xác: "__MISSING_DATA_TEXT__".

            ### 🔗 QUY ĐỊNH TRÍCH DẪN (CITATION):
            - Mọi thông tin quan trọng (con số, quy định bang, tên thẻ, thời hạn) phải có trích dẫn.
            - Định dạng trích dẫn bắt buộc: [Nguồn: Tiêu đề tài liệu | URL: đường_dẫn_nếu_có].
            - Ưu tiên dùng URL trong metadata của context.

            ### ✍️ PHONG CÁCH VĂN BẢN:
            - Ngôn ngữ mặc định: tiếng Việt (chỉ dùng tiếng Anh khi người dùng yêu cầu).
            - Văn phong: chuyên nghiệp, trung tính, đáng tin cậy.
            - Độc giả: nhà tuyển dụng, HR và người lao động tại Úc.

            ### 📋 CẤU TRÚC ĐẦU RA:
            Nếu người dùng KHÔNG yêu cầu định dạng đặc biệt, tuân thủ cấu trúc mặc định:
            1. Tiêu đề (H1) chuẩn SEO.
            2. Tóm tắt (Intro) ngắn gọn 1-2 câu.
            3. Các mục chính (H2, H3), phân loại theo bang/vùng lãnh thổ nếu có.
            4. Kết luận.
            5. Danh mục nguồn.
            
            🚨 [ĐẶC BIỆT]: Nếu có `{custom_instructions}`, BẠN PHẢI BỎ QUA cấu trúc mặc định bên trên và TUYỆT ĐỐI tuân thủ theo yêu cầu tuỳ chỉnh (Ví dụ: viết thơ, viết code, listicle, v.v.).

            ### 🔍 CHAIN OF VERIFICATION (CoVe):
            - Trước khi trả lời, rà soát lại toàn bài và xóa các câu không có bằng chứng trong context.
            - Nếu thiếu dữ liệu cho một ý bắt buộc, thay bằng: "__MISSING_DATA_TEXT__".

            Thông tin yêu cầu:
            - Topic: {topic}
            - Intent: {intent}
            - Audience: {audience}
            - Tone: {tone}
            - Length: {length}
            - Custom Instructions: {custom_instructions}

            Dữ liệu nội bộ đã truy xuất:
            {context}
            """.strip().replace("__MISSING_DATA_TEXT__", MISSING_INTERNAL_DATA_TEXT)
        self._prompt_template = PromptTemplate.from_template(system_prompt)
        self._generate_chain = RunnableLambda(self._generate)

    def runtime_status(self) -> dict[str, Any]:
        if settings.use_pgvector_retrieval and self._pgvector_connection_dsn() is not None:
            retrieval_mode = "pgvector"
        elif self._vector_store is not None and settings.use_pinecone_retrieval:
            retrieval_mode = "pinecone"
        else:
            retrieval_mode = "local"

        if self._agent_executor is not None and settings.use_agentic_rag:
            generation_mode = "agentic"
        elif self._llm is not None and settings.use_live_llm:
            generation_mode = "llm"
        else:
            generation_mode = "fallback"

        return {
            "retrieval_mode": retrieval_mode,
            "generation_mode": generation_mode,
            "quality_gate_enabled": settings.enforce_quality_gate,
            "quality_rules": {
                "min_sections": settings.min_sections,
                "min_sources_used": settings.min_sources_used,
                "min_draft_chars": settings.min_draft_chars,
            },
        }

    def _build_agent_executor(self) -> Any | None:
        if not settings.use_agentic_rag:
            return None
        if self._llm is None:
            return None

        try:
            agents_module = importlib.import_module("langchain.agents")
            tools_module = importlib.import_module("langchain.tools")
            initialize_agent = getattr(agents_module, "initialize_agent", None)
            agent_type = getattr(agents_module, "AgentType", None)
            tool_cls = getattr(tools_module, "Tool", None)
        except Exception:
            return None

        if initialize_agent is None or agent_type is None or tool_cls is None:
            return None

        tools = [
            tool_cls(
                name="database_vector_search",
                func=self._tool_database_vector_search,
                description=(
                    "Search PostgreSQL pgvector knowledge base for police check and compliance context. "
                    "Input should be a specific question."
                ),
            ),
            tool_cls(
                name="pinecone_search",
                func=self._tool_pinecone_search,
                description=(
                    "Search domain knowledge for police check, recruitment, compliance context. "
                    "Input should be a specific question."
                ),
            ),
            tool_cls(
                name="validex_website_reader",
                func=self._tool_validex_website_reader,
                description=(
                    "Read the configured Validex website page for pricing/service details. "
                    "Input should be a short note about what you need from the website."
                ),
            ),
            tool_cls(
                name="unsplash_image_search",
                func=self._tool_unsplash_image_search,
                description=(
                    "Search Unsplash for a section image using a concise keyword. "
                    "Input should be the keyword phrase and output includes image_url."
                ),
            ),
            tool_cls(
                name="seo_blog_check",
                func=self._tool_seo_blog_check,
                description=(
                    "Validate SEO quality of a blog draft. Input must be the current draft text."
                ),
            ),
            tool_cls(
                name="universal_web_scraper",
                func=self._tool_universal_web_scraper,
                description=(
                    "Scrape and extract clean text from ANY website URL. "
                    "Input MUST be a valid HTTP/HTTPS URL."
                ),
            ),
            tool_cls(
                name="image_ocr_extractor",
                func=self._tool_image_ocr_extractor,
                description=(
                    "Extract text (OCR) from an image URL using Google Gemini Multimodal. "
                    "Input MUST be a valid HTTP/HTTPS image URL."
                ),
            ),
        ]

        try:
            return initialize_agent(
                tools=tools,
                llm=self._llm,
                agent=agent_type.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=max(2, settings.agent_max_iterations),
            )
        except Exception:
            return None

    def _tool_database_vector_search(self, query: str) -> str:
        bundle = self._retrieve_from_pgvector(query, settings.top_k)
        if not bundle.documents:
            return "No relevant pgvector results found."
        lines = []
        for doc in bundle.documents[:3]:
            source_url = str(doc.metadata.get("source_url", "")).strip() or "n/a"
            page_ref = str(doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("chunk_id") or "n/a")
            lines.append(
                f"[doc={doc.metadata.get('doc_id', 'unknown_doc')} | url={source_url} | page={page_ref}] "
                f"score={doc.metadata.get('score', 0)}: {doc.page_content[:240]}"
            )
        return "\n".join(lines)

    def _tool_pinecone_search(self, query: str) -> str:
        bundle = self._retrieve_from_pinecone(query, settings.top_k)
        if not bundle.documents:
            return "No relevant vector results found."
        lines = []
        for doc in bundle.documents[:3]:
            source_url = str(doc.metadata.get("source_url", "")).strip() or "n/a"
            page_ref = str(doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("chunk_id") or "n/a")
            lines.append(
                f"[doc={doc.metadata.get('doc_id', 'unknown_doc')} | url={source_url} | page={page_ref}] "
                f"score={doc.metadata.get('score', 0)}: {doc.page_content[:240]}"
            )
        return "\n".join(lines)

    def _tool_validex_website_reader(self, _: str) -> str:
        if not settings.validex_website_url:
            return "VALIDEX_WEBSITE_URL is not configured."

        host = urlparse(settings.validex_website_url).netloc.lower()
        allowlist = {item.strip().lower() for item in settings.tool_allowed_domains.split(",") if item.strip()}
        if host not in allowlist:
            return "Configured website domain is not in TOOL_ALLOWED_DOMAINS."

        def _fetch() -> bytes:
            with urlopen(settings.validex_website_url, timeout=8) as response:  # noqa: S310
                return response.read()

        try:
            # Run blocking urlopen in a thread to avoid blocking the event loop.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    html_bytes = loop.run_in_executor(pool, _fetch)
                    # run_in_executor returns a Future; in a sync context we use the thread pool directly.
                    html_bytes = pool.submit(_fetch).result(timeout=10)
            else:
                html_bytes = _fetch()
            html = html_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return "Unable to fetch website content right now."

        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "Website content is empty after cleanup."
        return text[:2200]

    def _tool_universal_web_scraper(self, url: str) -> str:
        url = url.strip()
        if not url.startswith("http"):
            return "Invalid URL provided. Must start with http:// or https://"
        
        try:
            from curl_cffi import requests as cffi_requests
            from bs4 import BeautifulSoup
            response = cffi_requests.get(url, impersonate="chrome", timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            if not text:
                return "Scraping succeeded but no readable text found."
            return text[:6000]
        except Exception as e:
            return f"Scraping failed: {e}"

    def _tool_image_ocr_extractor(self, url: str) -> str:
        url = url.strip()
        if not url.startswith("http"):
            return "Invalid URL provided for OCR. Must start with http:// or https://"
        
        if self._llm is None:
            return "LLM is not configured, cannot perform OCR."
            
        try:
            import urllib.request
            import base64
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as response:
                image_data = response.read()
            
            # Determine mime type from URL extension or default to jpeg
            mime_type = "image/jpeg"
            if url.lower().endswith(".png"):
                mime_type = "image/png"
            elif url.lower().endswith(".webp"):
                mime_type = "image/webp"
                
            base64_img = base64.b64encode(image_data).decode("utf-8")
            
            from langchain_core.messages import HumanMessage
            msg = HumanMessage(content=[
                {"type": "text", "text": "Hãy trích xuất TẤT CẢ văn bản, chữ viết, bảng biểu (OCR) có trong hình ảnh này một cách chính xác nhất bằng tiếng Việt."},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}}
            ])
            
            ocr_response = self._llm.invoke([msg])
            result = getattr(ocr_response, "content", "")
            if not result:
                return "OCR completed but no text was detected."
            return str(result)
        except Exception as e:
            return f"OCR extraction failed: {e}"

    def _tool_unsplash_image_search(self, query: str) -> str:
        image_url, alt_text = self._search_unsplash_image(query)
        if not image_url:
            return "No Unsplash image found for this keyword."
        return json.dumps(
            {
                "keyword": query,
                "image_url": image_url,
                "alt_text": alt_text or query,
            },
            ensure_ascii=False,
        )

    def _tool_seo_blog_check(self, draft: str) -> str:
        text = draft.strip()
        if not text:
            return "SEO check: empty draft."

        words = len(text.split())
        has_h2 = "## " in text
        has_image = "![" in text and "](" in text
        bullets = text.count("- ")

        notes = []
        notes.append("SEO check result:")
        notes.append(f"- word_count: {words}")
        notes.append(f"- has_h2_headings: {'yes' if has_h2 else 'no'}")
        notes.append(f"- has_images: {'yes' if has_image else 'no'}")
        notes.append(f"- bullet_points: {bullets}")
        notes.append("- recommendation: include keyword naturally in first section and at least 3 headings")
        return "\n".join(notes)

    def _search_unsplash_image(self, keyword: str) -> tuple[str | None, str | None]:
        if not settings.use_unsplash_images:
            return (None, None)
        access_key = settings.unsplash_access_key.strip()
        if not access_key:
            return (None, None)

        clean_keyword = re.sub(r"\s+", " ", keyword).strip()
        if not clean_keyword:
            return (None, None)

        endpoint = (
            settings.unsplash_api_base.rstrip("/")
            + "/search/photos"
            + f"?query={quote_plus(clean_keyword[:120])}&per_page=1&orientation=landscape&content_filter=high"
        )
        request = Request(
            endpoint,
            headers={
                "Authorization": f"Client-ID {access_key}",
                "Accept-Version": "v1",
            },
        )
        try:
            with urlopen(request, timeout=max(3, settings.unsplash_timeout_seconds)) as response:
                payload = json.loads(response.read().decode("utf-8", errors="ignore"))
        except Exception:
            return (None, None)

        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return (None, None)

        first = results[0] if isinstance(results[0], dict) else {}
        urls = first.get("urls") if isinstance(first.get("urls"), dict) else {}
        image_url = urls.get("regular") or urls.get("full") or urls.get("small")
        if not image_url:
            return (None, None)

        alt_text = str(first.get("alt_description") or first.get("description") or "").strip() or None
        return (str(image_url), alt_text)

    def _resolve_section_image(self, topic: str, image_keyword: str, heading: str) -> tuple[str, str]:
        keyword = re.sub(r"\s+", " ", image_keyword).strip() or heading or topic
        # Allow LLM to explicitly remove an image by setting keyword to REMOVE_IMAGE.
        if keyword.upper() == "REMOVE_IMAGE":
            return ("", "")
        image_url, alt_text = self._search_unsplash_image(keyword)
        if image_url:
            return (image_url, alt_text or f"{heading} illustration")
        return (build_section_image_url(topic, keyword), f"{heading} illustration")

    @staticmethod
    def _normalize_parsed_prompt(
        prompt: str,
        intent: str,
        topic: str,
        tone: str,
        audience: str,
        length: str,
        custom_instructions: str = "",
    ) -> ParsedPrompt:
        intent_value = re.sub(r"\s+", "_", str(intent or "").strip().lower())
        if intent_value in {"rewrite", "edit", "revise", "update", "modify"}:
            intent_value = "rewrite"
        elif intent_value in {"shorten", "summary", "summarize", "condense"}:
            intent_value = "shorten"
        else:
            intent_value = "create_blog"

        topic_value = re.sub(r"\s+", " ", str(topic or "")).strip(" .,!?:;\n\t")
        if not topic_value:
            topic_value = "current draft" if intent_value in {"rewrite", "shorten"} else "general topic"

        tone_value = re.sub(r"\s+", "_", str(tone or "").strip().lower()) or "clear_professional"
        audience_value = re.sub(r"\s+", " ", str(audience or "").strip().lower()) or "general audience"

        length_value = str(length or "").strip().lower()
        if length_value not in {"short", "medium", "long"}:
            length_value = "medium"

        custom_instructions_value = str(custom_instructions or "").strip()

        return ParsedPrompt(
            raw_prompt=prompt,
            intent=intent_value,
            topic=topic_value,
            tone=tone_value,
            audience=audience_value,
            length=length_value,
            custom_instructions=custom_instructions_value,
        )

    def _classify_llm_error(self, error_text: str) -> str:
        normalized = str(error_text or "").upper()
        if "RESOURCE_EXHAUSTED" in normalized or "QUOTA" in normalized or "RATE LIMIT" in normalized or "429" in normalized:
            return "quota_exhausted"
        if "UNAUTHENTICATED" in normalized or "AUTH" in normalized or "API KEY" in normalized or "401" in normalized:
            return "auth_error"
        if "TIMEOUT" in normalized or "DEADLINE" in normalized or "504" in normalized:
            return "timeout"
        if "INVALID_ARGUMENT" in normalized or "400" in normalized:
            return "invalid_request"
        return "invoke_error"

    def _record_llm_failure(self, llm_trace: dict[str, Any] | None, stage: str, error_value: str) -> None:
        if not isinstance(llm_trace, dict):
            return
        llm_trace["attempted"] = True
        failures = llm_trace.setdefault("failures", [])
        if not isinstance(failures, list):
            llm_trace["failures"] = []
            failures = llm_trace["failures"]

        message = self._normalize_whitespace_for_log(str(error_value or ""), max_chars=480)
        failures.append(
            {
                "stage": stage,
                "reason": self._classify_llm_error(message),
                "message": message,
            }
        )
        if len(failures) > MAX_LLM_FAILURE_RECORDS:
            del failures[0 : len(failures) - MAX_LLM_FAILURE_RECORDS]
        self._cb_record_failure()

    def _cb_is_open(self) -> bool:
        """Return True if the circuit breaker is open (LLM should be skipped)."""
        import time as _time
        if self._cb_open_until > 0 and _time.time() < self._cb_open_until:
            logger.warning("circuit_breaker.open — skipping LLM call for cooldown")
            return True
        if self._cb_open_until > 0 and _time.time() >= self._cb_open_until:
            # Cooldown expired, half-open: allow one attempt
            self._cb_open_until = 0.0
            self._cb_consecutive_failures = 0
        return False

    def _cb_record_success(self) -> None:
        self._cb_consecutive_failures = 0
        self._cb_open_until = 0.0

    def _cb_record_failure(self) -> None:
        import time as _time
        self._cb_consecutive_failures += 1
        threshold = max(1, settings.circuit_breaker_threshold)
        if self._cb_consecutive_failures >= threshold:
            cooldown = max(10, settings.circuit_breaker_cooldown_seconds)
            self._cb_open_until = _time.time() + cooldown
            logger.warning(
                "circuit_breaker.tripped — LLM disabled for %ds after %d consecutive failures",
                cooldown, self._cb_consecutive_failures,
            )

    def _parse_with_llm(self, prompt: str, llm_trace: dict[str, Any] | None = None) -> ParsedPrompt | None:
        if self._llm is None:
            return None
        if not hasattr(self._llm, "with_structured_output"):
            return None

        if isinstance(llm_trace, dict):
            llm_trace["attempted"] = True

        instruction = (
            "Extract prompt metadata for a blog system.\n"
            "Return fields: intent, topic, tone, audience, length, custom_instructions.\n"
            "Rules:\n"
            "- intent must be one of: create_blog, rewrite, shorten.\n"
            "- If the user prompt is just a keyword, short phrase, or question WITHOUT a clear edit/rewrite/shorten command, "
            "assume intent=create_blog and use the EXACT user input as the topic.\n"
            "- If user asks to edit/rewrite/update an existing draft (including image count changes, removing images, etc.), "
            "use intent=rewrite and topic=current draft unless a new explicit topic is provided.\n"
            "- If user asks to make content shorter/summarize, use intent=shorten and topic=current draft unless a new explicit topic is provided.\n"
            "- length must be one of: short, medium, long. Infer from explicit word targets when possible.\n"
            "- audience should capture explicit target audience if present, otherwise general audience.\n"
            "- tone should be concise and normalized (for example: professional, friendly, casual, clear_professional).\n"
            "- custom_instructions: Extract ANY special structural, stylistic, formatting rules or non-standard requests "
            "(e.g., 'viết 1 bài thơ', 'không dùng markdown', 'chỉ trả lời 1 câu', 'dạng danh sách', 'thêm icon'). "
            "Leave empty if it's a standard blog request.\n"
            f"User prompt:\n{prompt}"
        )

        try:
            parser_llm = self._llm.with_structured_output(PromptParseSchema)
            result = parser_llm.invoke(instruction)
        except Exception as exc:
            self._record_llm_failure(llm_trace, "parse", str(exc))
            self._log_raw_llm_exchange("parse_error", instruction, f"ERROR: {exc}")
            return None

        try:
            if isinstance(result, PromptParseSchema):
                parsed_payload = result
            elif isinstance(result, dict):
                parsed_payload = PromptParseSchema.model_validate(result)
            elif hasattr(result, "model_dump"):
                parsed_payload = PromptParseSchema.model_validate(result.model_dump())
            else:
                return None
        except ValidationError:
            return None

        return self._normalize_parsed_prompt(
            prompt=prompt,
            intent=parsed_payload.intent,
            topic=parsed_payload.topic,
            tone=parsed_payload.tone,
            audience=parsed_payload.audience,
            length=parsed_payload.length,
            custom_instructions=getattr(parsed_payload, "custom_instructions", ""),
        )

    @staticmethod
    def _normalize_whitespace_for_log(value: str, max_chars: int = 7000) -> str:
        compact = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars] + " ...[truncated]"

    def _log_raw_llm_exchange(self, stage: str, prompt_text: str, response_text: str) -> None:
        # Enabled by default for debugging prompt/response mismatches in production-like flows.
        if os.getenv("LOG_RAW_LLM_IO", "1") != "1":
            return
        logger.debug(
            "LLM_RAW_PROMPT[%s]: %s",
            stage,
            self._normalize_whitespace_for_log(prompt_text),
        )
        logger.debug(
            "LLM_RAW_RESPONSE[%s]: %s",
            stage,
            self._normalize_whitespace_for_log(response_text),
        )

    def _parse(self, payload: dict) -> ParsedPrompt:
        prompt = payload["prompt"]
        llm_trace = payload.get("llm_trace")
        parsed = self._parse_with_llm(prompt, llm_trace=llm_trace)
        if parsed is None:
            if isinstance(llm_trace, dict):
                llm_trace["parse_mode"] = "heuristic"
            parsed = parse_user_input(prompt)
        elif isinstance(llm_trace, dict):
            llm_trace["parse_mode"] = "llm"

        # Guardrail: image-count prompts are follow-up edits, even if LLM parsing is noisy.
        if extract_requested_image_limit(prompt) is not None and parsed.intent == "create_blog":
            return replace(parsed, intent="rewrite", topic="current draft")

        return parsed

    def _build_llm(self) -> Any | None:
        if not settings.use_live_llm:
            return None

        provider = settings.llm_provider.strip().lower()
        if provider not in {"auto", "openai", "google"}:
            provider = "auto"

        preferred_google_model = settings.google_model_name.strip()
        if not preferred_google_model:
            preferred_google_model = settings.model_name.strip() or "gemini-2.5-flash"
        if not preferred_google_model.startswith("models/"):
            preferred_google_model = f"models/{preferred_google_model}"

        if provider in {"auto", "google"} and settings.google_api_key and ChatGoogleGenerativeAI is not None:
            try:
                return ChatGoogleGenerativeAI(
                    model=preferred_google_model,
                    google_api_key=settings.google_api_key,
                    temperature=0.2,
                    max_retries=1,
                    timeout=30.0,
                )
            except Exception:
                if provider == "google":
                    return None

        if provider in {"auto", "openai"} and settings.openai_api_key and ChatOpenAI is not None:
            try:
                return ChatOpenAI(
                    model=settings.model_name,
                    api_key=settings.openai_api_key,
                    temperature=0.2,
                    max_retries=1,
                    timeout=30.0,
                )
            except Exception:
                return None

        return None

    def _build_embedding_model(self) -> Any | None:
        provider = settings.embedding_provider.strip().lower()
        if provider not in {"auto", "openai", "google"}:
            provider = "auto"

        google_output_dimensionality: int | None = None
        raw_google_dim = os.getenv("GOOGLE_EMBEDDING_OUTPUT_DIMENSION", "").strip()
        if raw_google_dim:
            try:
                google_output_dimensionality = int(raw_google_dim)
            except ValueError:
                google_output_dimensionality = None
        if google_output_dimensionality is None:
            # Keep query embedding dimension aligned with current pgvector table default.
            google_output_dimensionality = 1536

        # Prioritize explicit Google embedding model when using Google provider.
        preferred_google_embedding = settings.google_embedding_model.strip()
        if not preferred_google_embedding:
            fallback_google_embedding = settings.embedding_model.strip()
            if fallback_google_embedding.startswith("models/"):
                preferred_google_embedding = fallback_google_embedding
        if not preferred_google_embedding:
            preferred_google_embedding = "models/gemini-embedding-001"
        if not preferred_google_embedding.startswith("models/"):
            preferred_google_embedding = f"models/{preferred_google_embedding}"

        if provider in {"auto", "google"} and settings.google_api_key and GoogleGenerativeAIEmbeddings is not None:
            try:
                return GoogleGenerativeAIEmbeddings(
                    model=preferred_google_embedding,
                    google_api_key=settings.google_api_key,
                    output_dimensionality=google_output_dimensionality,
                )
            except Exception:
                if provider == "google":
                    return None

        if provider in {"auto", "openai"} and settings.openai_api_key and OpenAIEmbeddings is not None:
            try:
                return OpenAIEmbeddings(
                    model=settings.embedding_model,
                    api_key=settings.openai_api_key,
                )
            except Exception:
                return None

        return None

    def _build_vector_store(self) -> Any | None:
        if not settings.use_pinecone_retrieval:
            return None
        if not (settings.pinecone_api_key and settings.pinecone_index):
            return None
        if self._embedding_model is None:
            return None

        try:
            pinecone_module = importlib.import_module("langchain_pinecone")
            pinecone_vector_store_cls = getattr(pinecone_module, "PineconeVectorStore", None)
        except Exception:
            pinecone_vector_store_cls = None

        if pinecone_vector_store_cls is None:
            return None

        try:
            return pinecone_vector_store_cls(
                index_name=settings.pinecone_index,
                embedding=self._embedding_model,
                namespace=settings.pinecone_namespace or None,
                pinecone_api_key=settings.pinecone_api_key,
                text_key="text",
            )
        except TypeError:
            try:
                return pinecone_vector_store_cls.from_existing_index(
                    index_name=settings.pinecone_index,
                    embedding=self._embedding_model,
                    namespace=settings.pinecone_namespace or None,
                    text_key="text",
                    pinecone_api_key=settings.pinecone_api_key,
                )
            except Exception:
                return None
        except Exception:
            return None

    def _pgvector_connection_dsn(self) -> str | None:
        raw = os.getenv("DATABASE_URL", "").strip()
        if raw:
            return raw
        alt = os.getenv("PGVECTOR_CONNECTION_STRING", "").strip()
        if alt.startswith("postgresql+psycopg2://"):
            return "postgresql://" + alt.split("postgresql+psycopg2://", 1)[1]
        return alt or None

    @staticmethod
    def _vector_literal(values: list[float]) -> str:
        return "[" + ",".join(f"{float(item):.8f}" for item in values) + "]"

    def _query_embedding(self, query: str) -> list[float] | None:
        if self._embedding_model is not None and hasattr(self._embedding_model, "embed_query"):
            try:
                values = self._embedding_model.embed_query(query)
                return [float(item) for item in values]
            except Exception:
                pass

        if settings.allow_fake_embeddings:
            dim = max(1, min(8192, settings.fake_embedding_dim))
            seed = int(hashlib.sha1(query.encode("utf-8")).hexdigest()[:16], 16)
            rng = random.Random(seed)
            return [rng.uniform(-1.0, 1.0) for _ in range(dim)]

        return None

    @staticmethod
    def _normalize_score(score: float | int | None) -> tuple[int, float]:
        if score is None:
            return (0, 0.0)
        raw = float(score)
        if raw <= 1.0:
            return (int(round(raw * 100)), max(0.0, min(1.0, raw)))
        return (int(round(raw)), min(1.0, raw / 100.0))

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return 0
        return max(1, int(math.ceil(len(compact) / 4)))

    def _token_budget_plan(self, length: str) -> TokenBudgetPlan:
        length_key = length if length in {"short", "medium", "long"} else "medium"

        if length_key == "short":
            output_tokens = settings.output_tokens_short
            range_min = settings.top_k_short_min
            range_max = settings.top_k_short_max
        elif length_key == "long":
            output_tokens = settings.output_tokens_long
            range_min = settings.top_k_long_min
            range_max = settings.top_k_long_max
        else:
            output_tokens = settings.output_tokens_medium
            range_min = settings.top_k_medium_min
            range_max = settings.top_k_medium_max

        ratio_min = max(1.0, settings.input_output_ratio_min)
        ratio_max = max(ratio_min, settings.input_output_ratio_max)

        input_tokens_min = int(math.ceil(output_tokens * ratio_min))
        input_tokens_max = int(math.ceil(output_tokens * ratio_max))
        input_tokens_target = int(round((input_tokens_min + input_tokens_max) / 2))

        chunk_tokens = max(1, settings.chunk_token_estimate)
        required_chunks = int(math.ceil(input_tokens_target / chunk_tokens))

        top_k_floor = max(1, min(range_min, range_max))
        top_k_ceil = max(top_k_floor, max(range_min, range_max))
        recommended_top_k = min(max(required_chunks, top_k_floor), top_k_ceil)

        return TokenBudgetPlan(
            length_profile=length_key,
            output_tokens=output_tokens,
            input_tokens_min=input_tokens_min,
            input_tokens_target=input_tokens_target,
            input_tokens_max=input_tokens_max,
            recommended_top_k=recommended_top_k,
        )

    def _select_context_documents(
        self,
        documents: list[Document],
        token_plan: TokenBudgetPlan,
    ) -> tuple[list[Document], int]:
        if not documents:
            return [], 0

        selected: list[Document] = []
        total_tokens = 0
        for doc in documents:
            estimate = self._estimate_token_count(doc.page_content)
            if selected and total_tokens + estimate > token_plan.input_tokens_max:
                break
            selected.append(doc)
            total_tokens += estimate
            if total_tokens >= token_plan.input_tokens_target:
                break

        if not selected:
            first = documents[0]
            selected = [first]
            total_tokens = self._estimate_token_count(first.page_content)

        return selected, total_tokens

    @staticmethod
    def _get_doc_id(doc: Document, fallback_id: str) -> str:
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("id")
        if not doc_id:
            source = str(doc.metadata.get("source", ""))
            stem = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            doc_id = stem.replace(".txt", "") if stem else fallback_id
        return str(doc_id)

    def _retrieve_from_pinecone(self, query: str, top_k: int) -> RetrievalBundle:
        if self._vector_store is None:
            return self._retrieve_from_local_guard(query, top_k)

        docs_with_scores: list[tuple[Document, float]] = []
        try:
            if hasattr(self._vector_store, "similarity_search_with_relevance_scores"):
                raw_results = self._vector_store.similarity_search_with_relevance_scores(query, k=top_k)
                docs_with_scores = [(doc, float(score)) for doc, score in raw_results]
            elif hasattr(self._vector_store, "similarity_search_with_score"):
                raw_results = self._vector_store.similarity_search_with_score(query, k=top_k)
                docs_with_scores = [(doc, float(score)) for doc, score in raw_results]
            else:
                docs = self._vector_store.similarity_search(query, k=top_k)
                docs_with_scores = [(doc, 0.6) for doc in docs]
        except Exception:
            return self._retrieve_from_local_guard(query, top_k)

        if not docs_with_scores:
            return RetrievalBundle(
                decision=RetrievalDecision([], "no_match", 0.0, 0, "no vector match"),
                documents=[],
            )

        top_display_score, top_confidence = self._normalize_score(docs_with_scores[0][1])
        retrieved_docs: list[Document] = []
        for index, (doc, score) in enumerate(docs_with_scores):
            display_score, semantic_score = self._normalize_score(score)
            merged_metadata = dict(doc.metadata)
            merged_metadata["doc_id"] = self._get_doc_id(doc, f"vector_doc_{index + 1}")
            merged_metadata["score"] = display_score
            merged_metadata["semantic_score"] = semantic_score
            retrieved_docs.append(Document(page_content=doc.page_content, metadata=merged_metadata))

        decision = RetrievalDecision(
            docs=[
                RetrievedDoc(
                    doc_id=str(doc.metadata.get("doc_id", f"vector_doc_{idx + 1}")),
                    score=int(doc.metadata.get("score", 0)),
                    content=doc.page_content,
                    semantic_score=float(doc.metadata.get("semantic_score", 0.0)),
                )
                for idx, doc in enumerate(retrieved_docs)
            ],
            status="ok",
            confidence=top_confidence,
            top_score=top_display_score,
            reason="vector retrieval successful",
        )
        return RetrievalBundle(decision=decision, documents=retrieved_docs)

    def _retrieve_from_pgvector(self, query: str, top_k: int) -> RetrievalBundle:
        dsn = self._pgvector_connection_dsn()
        if not dsn:
            return self._retrieve_from_local_guard(query, top_k)

        query_vector = self._query_embedding(query)
        if not query_vector:
            return self._retrieve_from_local_guard(query, top_k)

        vector_literal = self._vector_literal(query_vector)
        embedding_provider_filter = ""
        if settings.pgvector_require_non_fake_embeddings:
            embedding_provider_filter = "and coalesce(embedding_provider, 'unknown') != 'fake'"

        try:
            with psycopg.connect(dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        select
                            chunk_id,
                            doc_id,
                            content,
                            source_url,
                            source_domain,
                            source_type,
                            topic,
                            region,
                            title,
                            authority_score,
                            approved,
                            1 - (embedding <=> %s::vector) as similarity
                        from {settings.pgvector_table}
                                                where approved = true
                                                    {embedding_provider_filter}
                        order by embedding <=> %s::vector
                        limit %s
                        """,
                        (vector_literal, vector_literal, top_k),
                    )
                    rows = cur.fetchall()
        except Exception:
            return self._retrieve_from_local_guard(query, top_k)

        if not rows:
            return RetrievalBundle(
                decision=RetrievalDecision([], "no_match", 0.0, 0, "no pgvector match"),
                documents=[],
            )

        retrieved_docs: list[Document] = []
        scores: list[float] = []
        for row in rows:
            similarity = float(row[11] or 0.0)
            if similarity < settings.pgvector_min_similarity:
                continue
            authority_score = float(row[9] or 0.0)
            blended = max(0.0, similarity + (authority_score * 0.08))
            score = int(round(min(1.0, blended) * 100))
            scores.append(blended)
            retrieved_docs.append(
                Document(
                    page_content=str(row[2] or ""),
                    metadata={
                        "chunk_id": str(row[0] or ""),
                        "doc_id": str(row[1] or "unknown_doc"),
                        "source_url": str(row[3] or ""),
                        "source_domain": str(row[4] or ""),
                        "source_type": str(row[5] or ""),
                        "topic": str(row[6] or ""),
                        "region": str(row[7] or "AU"),
                        "title": str(row[8] or "Untitled"),
                        "authority_score": authority_score,
                        "approved": bool(row[10]),
                        "score": score,
                        "semantic_score": round(similarity, 4),
                    },
                )
            )

        if not retrieved_docs:
            return RetrievalBundle(
                decision=RetrievalDecision([], "low_confidence", 0.0, 0, "pgvector matches under similarity threshold"),
                documents=[],
            )

        top_score = int(retrieved_docs[0].metadata.get("score", 0))
        confidence = max(0.0, min(1.0, scores[0])) if scores else 0.0

        decision = RetrievalDecision(
            docs=[
                RetrievedDoc(
                    doc_id=str(doc.metadata.get("doc_id", "unknown_doc")),
                    score=int(doc.metadata.get("score", 0)),
                    content=doc.page_content,
                    semantic_score=float(doc.metadata.get("semantic_score", 0.0)),
                )
                for doc in retrieved_docs
            ],
            status="ok",
            confidence=confidence,
            top_score=top_score,
            reason="pgvector retrieval successful",
        )
        return RetrievalBundle(decision=decision, documents=retrieved_docs)

    def _retrieve_from_local_guard(self, query: str, top_k: int) -> RetrievalBundle:
        decision = retrieve_with_guard(
            query,
            settings.data_processed_dir,
            top_k,
            settings.metadata_path,
            settings.min_top_score,
            settings.min_confidence,
        )
        docs = [
            Document(page_content=doc.content, metadata={"doc_id": doc.doc_id, "score": doc.score, "semantic_score": doc.semantic_score})
            for doc in decision.docs
        ]
        return RetrievalBundle(decision=decision, documents=docs)

    def _retrieve(self, payload: dict) -> RetrievalBundle:
        topic = payload["effective_topic"]
        retrieval_top_k = int(payload.get("retrieval_top_k") or settings.top_k)
        retrieval_top_k = max(1, min(20, retrieval_top_k))
        
        # Increase initial top_k for reranking buffer
        initial_top_k = retrieval_top_k * 3

        logger.info("pipeline.retrieve_start", extra={"topic": topic, "top_k": retrieval_top_k, "initial_top_k": initial_top_k})
        if settings.use_pgvector_retrieval and self._pgvector_connection_dsn() is not None:
            bundle = self._retrieve_from_pgvector(topic, initial_top_k)
            if bundle.decision.status in {"low_confidence", "no_match"} and not bundle.documents:
                if not settings.pgvector_require_non_fake_embeddings:
                    local_bundle = self._retrieve_from_local_guard(topic, initial_top_k)
                    if local_bundle.decision.status == "ok" and local_bundle.documents:
                        bundle = local_bundle
        elif settings.use_pinecone_retrieval:
            bundle = self._retrieve_from_pinecone(topic, initial_top_k)
        else:
            bundle = self._retrieve_from_local_guard(topic, initial_top_k)
            
        # Rerank with FlashRank
        if len(bundle.documents) > retrieval_top_k:
            try:
                from flashrank import Ranker, RerankRequest
                ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="data/models")
                passages = []
                for i, doc in enumerate(bundle.documents):
                    passages.append({
                        "id": i,
                        "text": doc.page_content,
                        "meta": doc.metadata
                    })
                rerankrequest = RerankRequest(query=topic, passages=passages)
                results = ranker.rerank(rerankrequest)
                
                reranked_docs = []
                for res in results[:retrieval_top_k]:
                    reranked_docs.append(bundle.documents[res["id"]])
                
                bundle.documents = reranked_docs
            except Exception as exc:
                logger.warning("Reranking failed: %s, falling back to top_k slicing", exc)
                bundle.documents = bundle.documents[:retrieval_top_k]
                
        # Autonomous Web Search Fallback
        if len(bundle.documents) == 0 or bundle.decision.status in {"no_match", "low_confidence", "out_of_domain"}:
            logger.info("pipeline.autonomous_web_search", extra={"topic": topic})
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    ddg_results = list(ddgs.text(topic, max_results=3))
                
                if ddg_results:
                    bundle.documents = [
                        Document(
                            page_content=r.get("body", ""),
                            metadata={
                                "doc_id": f"web_{i}",
                                "title": r.get("title", ""),
                                "source_url": r.get("href", ""),
                                "score": 100,
                                "semantic_score": 1.0,
                            }
                        )
                        for i, r in enumerate(ddg_results)
                    ]
                    bundle.decision.status = "web_search"
                    bundle.decision.message = "Fallback to Web Search successful"
                    logger.info("pipeline.autonomous_web_search_success", extra={"results": len(ddg_results)})
            except Exception as exc:
                logger.error("DuckDuckGo search failed: %s", exc)

        logger.info(
            "pipeline.retrieve_done",
            extra={
                "topic": topic,
                "top_k": retrieval_top_k,
                "status": bundle.decision.status,
                "confidence": round(bundle.decision.confidence, 4),
                "docs": len(bundle.documents),
            },
        )
        return bundle

    def _quality_gate_result(self, generated: GeneratedBlog) -> tuple[bool, str]:
        if not settings.enforce_quality_gate:
            return True, "quality gate disabled"

        if len(generated.sections) < max(1, settings.min_sections):
            return False, "insufficient sections"
        if len(generated.sources_used) < max(0, settings.min_sources_used):
            return False, "insufficient source references"
        if len(generated.draft.strip()) < max(100, settings.min_draft_chars):
            return False, "draft length under threshold"
        return True, "quality gate passed"

    def _generate_with_fallback(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
        previous_draft: str | None,
    ) -> GeneratedBlog:
        retrieved_docs = [
            RetrievedDoc(
                doc_id=str(doc.metadata.get("doc_id", "unknown_doc")),
                score=int(doc.metadata.get("score", 0)),
                content=doc.page_content,
                semantic_score=float(doc.metadata.get("semantic_score", 0.0)),
            )
            for doc in docs
        ]
        return generate_blog_output(parsed, retrieved_docs, previous_draft=previous_draft)

    @staticmethod
    def _generated_to_payload(generated: GeneratedBlog) -> dict[str, Any]:
        return {
            "title": generated.title,
            "outline": generated.outline,
            "draft": generated.draft,
            "sources_used": generated.sources_used,
            "sections": [
                {
                    "heading": section.heading,
                    "body": section.body,
                    "image_url": section.image_url,
                    "image_alt": section.image_alt,
                }
                for section in generated.sections
            ],
        }

    def _generate_with_hybrid_fallback(
        self,
        parsed: ParsedPrompt,
        previous_draft: str | None,
        retrieval_status: str,
    ) -> GeneratedBlog:
        warning = settings.hybrid_warning_text.strip()

        # Try structured output first so the blog gets full sections + Unsplash images
        # even when RAG context is unavailable.
        if self._llm is not None and settings.use_live_llm:
            structured_result = self._generate_with_structured_output(
                parsed, docs=[], previous_draft=previous_draft,
            )
            if structured_result is not None:
                structured_result.sources_used = []
                if not structured_result.draft.startswith(warning):
                    structured_result.draft = f"{warning}\n\n{structured_result.draft}"
                return structured_result

            # Fallback: try markdown-direct generation.
            md_result = self._generate_markdown_directly_with_llm(
                parsed, docs=[], previous_draft=previous_draft,
            )
            if md_result is not None:
                md_result.sources_used = []
                if not md_result.draft.startswith(warning):
                    md_result.draft = f"{warning}\n\n{md_result.draft}"
                return md_result

        # Final fallback: static template.
        generated = self._generate_with_fallback(parsed, [], previous_draft)
        generated.sources_used = []
        if not generated.draft.startswith(warning):
            generated.draft = f"{warning}\n\n{generated.draft}"
        return generated

    @staticmethod
    def _trim_markdown_images(draft: str, image_limit: int, allowed_image_urls: set[str] | None = None) -> str:
        pattern = re.compile(r"^\s*!\[[^\]]*\]\(([^)]+)\)\s*$")
        lines = draft.splitlines()
        kept = 0
        output_lines: list[str] = []
        for line in lines:
            match = pattern.match(line)
            if match:
                image_url = match.group(1).strip()
                if allowed_image_urls is None:
                    if kept < image_limit:
                        output_lines.append(line)
                    kept += 1
                    continue

                keep_by_url = image_url in allowed_image_urls
                if keep_by_url and kept < image_limit:
                    output_lines.append(line)
                    kept += 1
                continue
            output_lines.append(line)
        return "\n".join(output_lines).strip()

    @staticmethod
    def _is_context_section_heading(heading: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(heading or "")).strip().lower()
        return normalized == "context from previous draft"

    def _apply_prompt_edit_constraints(self, prompt: str, generated_payload: dict[str, Any]) -> dict[str, Any]:
        image_limit = extract_requested_image_limit(prompt)
        if image_limit is None:
            return generated_payload

        allowed_image_urls: set[str] = set()
        sections = generated_payload.get("sections")
        if isinstance(sections, list):
            editable_sections: list[dict[str, Any]] = []
            for section in sections:
                if not isinstance(section, dict):
                    continue
                if self._is_context_section_heading(str(section.get("heading", ""))):
                    section["image_url"] = ""
                    continue
                editable_sections.append(section)

            for index, section in enumerate(editable_sections):
                if index >= image_limit:
                    section["image_url"] = ""

            for section in sections:
                if not isinstance(section, dict):
                    continue
                image_url = str(section.get("image_url", "")).strip()
                if image_url:
                    allowed_image_urls.add(image_url)

        draft = generated_payload.get("draft")
        if isinstance(draft, str) and draft:
            generated_payload["draft"] = self._trim_markdown_images(
                draft,
                image_limit,
                allowed_image_urls=allowed_image_urls,
            )

        return generated_payload

    @staticmethod
    def _docs_to_retrieved(docs: list[Document]) -> list[RetrievedDoc]:
        return [
            RetrievedDoc(
                doc_id=str(doc.metadata.get("doc_id", "unknown_doc")),
                score=int(doc.metadata.get("score", 0)),
                content=doc.page_content,
                semantic_score=float(doc.metadata.get("semantic_score", 0.0)),
            )
            for doc in docs
        ]

    @staticmethod
    def _doc_reference_line(doc: Document) -> str:
        doc_id = str(doc.metadata.get("doc_id", "unknown_doc"))
        title = str(doc.metadata.get("title", "")).strip() or doc_id
        source_url = str(doc.metadata.get("source_url", "")).strip() or "khong_co"
        return f"{title} | URL: {source_url}"

    @classmethod
    def _build_source_reference_list(cls, docs: list[Document], limit: int = 3) -> list[str]:
        references: list[str] = []
        seen: set[str] = set()
        capped_limit = max(1, limit)

        for doc in docs:
            reference = cls._doc_reference_line(doc)
            if reference in seen:
                continue
            references.append(reference)
            seen.add(reference)
            if len(references) >= capped_limit:
                break
        return references

    @staticmethod
    def _remove_source_lines(text: str) -> str:
        prefix = SOURCE_LINE_PREFIX.lower()
        lines = [
            line
            for line in text.splitlines()
            if not line.strip().lower().startswith(prefix)
            and not line.strip().lower().startswith("[nguồn:")
        ]
        return "\n".join(lines).strip()

    def _enforce_grounding_and_citations(
        self,
        generated: GeneratedBlog,
        docs: list[Document],
    ) -> GeneratedBlog:
        references = self._build_source_reference_list(docs)
        citation_tokens = [f"[Nguồn: {reference}]" for reference in references]
        citation_line = " ".join(citation_tokens) if citation_tokens else ""

        normalized_sections: list[GeneratedBlog.Section] = []
        for section in generated.sections:
            base_body = self._remove_source_lines(str(section.body or ""))
            if not base_body:
                base_body = MISSING_INTERNAL_DATA_TEXT

            if citation_line:
                body = f"{base_body}\n\n{citation_line}".strip()
            else:
                body = base_body
                if MISSING_INTERNAL_DATA_TEXT not in body:
                    body = f"{body}\n\n{MISSING_INTERNAL_DATA_TEXT}".strip()

            normalized_sections.append(
                GeneratedBlog.Section(
                    heading=section.heading,
                    body=body,
                    image_url=section.image_url,
                    image_alt=section.image_alt,
                )
            )

        draft = str(generated.draft or "").strip()
        if not draft:
            draft = render_markdown_blog(generated.title, normalized_sections)

        if references:
            sources_block = SOURCES_SECTION_HEADING + "\n" + "\n".join(f"- [Nguồn: {reference}]" for reference in references)
            if SOURCES_SECTION_HEADING not in draft and "## Sources" not in draft:
                draft = f"{draft}\n\n{sources_block}".strip()
        elif MISSING_INTERNAL_DATA_TEXT not in draft:
            draft = f"{draft}\n\n{MISSING_INTERNAL_DATA_TEXT}".strip()

        sources_used = [str(doc.metadata.get("doc_id", "unknown_doc")) for doc in docs]
        if not sources_used:
            sources_used = list(dict.fromkeys(generated.sources_used))

        return GeneratedBlog(
            title=generated.title,
            outline=generated.outline,
            draft=draft,
            sources_used=list(dict.fromkeys(sources_used)),
            sections=normalized_sections,
        )

    @staticmethod
    def _extract_json_block(raw_text: str) -> dict[str, Any] | None:
        raw_text = raw_text.strip()
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", raw_text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _render_structured_markdown(
        title: str,
        introduction: str,
        sections: list[GeneratedBlog.Section],
        conclusion: str,
        meta_tags: str,
    ) -> str:
        blocks: list[str] = [f"# {title}"]

        intro_text = introduction.strip()
        if intro_text:
            blocks.extend(["", "## Introduction", "", intro_text])

        for section in sections:
            blocks.extend(
                [
                    "",
                    f"## {section.heading}",
                    "",
                    f"![{section.image_alt}]({section.image_url})",
                    "",
                    section.body,
                ]
            )

        conclusion_text = conclusion.strip()
        if conclusion_text:
            blocks.extend(["", "## Conclusion", "", conclusion_text])

        tags_text = meta_tags.strip()
        if tags_text:
            blocks.extend(["", f"<!-- meta_tags: {tags_text} -->"])

        return "\n".join(blocks).strip()

    @staticmethod
    def _generated_from_markdown(
        markdown_text: str,
        parsed: ParsedPrompt,
        docs: list[Document],
    ) -> GeneratedBlog:
        draft = str(markdown_text or "").strip()
        if not draft:
            raise ValueError("markdown draft is empty")

        title_match = re.search(r"(?m)^\s*#\s+(.+?)\s*$", draft)
        title = title_match.group(1).strip() if title_match else format_title(parsed.topic)

        heading_matches = list(re.finditer(r"(?m)^\s*##\s+(.+?)\s*$", draft))
        sections: list[GeneratedBlog.Section] = []
        for index, match in enumerate(heading_matches):
            heading = match.group(1).strip()
            start = match.end()
            end = heading_matches[index + 1].start() if index + 1 < len(heading_matches) else len(draft)
            section_block = draft[start:end].strip()
            if not heading or not section_block:
                continue

            image_match = re.search(r"!\[([^\]]*)\]\(([^)]+)\)", section_block)
            if image_match:
                image_alt = str(image_match.group(1) or "").strip() or f"{heading} illustration"
                image_url = str(image_match.group(2) or "").strip() or build_section_image_url(parsed.topic, heading)
            else:
                image_alt = f"{heading} illustration"
                image_url = build_section_image_url(parsed.topic, heading)

            sections.append(
                GeneratedBlog.Section(
                    heading=heading,
                    body=section_block,
                    image_url=image_url,
                    image_alt=image_alt,
                )
            )

        if not sections:
            fallback_heading = "Main Content"
            sections = [
                GeneratedBlog.Section(
                    heading=fallback_heading,
                    body=draft,
                    image_url=build_section_image_url(parsed.topic, fallback_heading),
                    image_alt=f"{fallback_heading} illustration",
                )
            ]

        sources_used = [str(doc.metadata.get("doc_id", "unknown_doc")) for doc in docs]
        outline = [section.heading for section in sections]
        return GeneratedBlog(
            title=title,
            outline=outline,
            draft=draft,
            sources_used=sources_used,
            sections=sections,
        )

    def _generate_markdown_directly_with_llm(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
        previous_draft: str | None,
        llm_trace: dict[str, Any] | None = None,
    ) -> GeneratedBlog | None:
        if self._llm is None:
            return None

        if isinstance(llm_trace, dict):
            llm_trace["attempted"] = True

        prompt_text = self._prompt_template.format(
            topic=parsed.topic,
            intent=parsed.intent,
            audience=parsed.audience,
            tone=parsed.tone,
            length=parsed.length,
            context=self._format_context(docs),
            custom_instructions=parsed.custom_instructions,
        )

        previous_note = ""
        if previous_draft:
            previous_note = (
                "\n\nPrevious draft context (truncated):\n"
                f"{previous_draft[:500]}"
            )

        llm_instruction = (
            prompt_text
            + previous_note
            + "\n\nYêu cầu bắt buộc: Trả về DUY NHẤT một bài blog Markdown hoàn chỉnh."
            + " Không trả JSON, không trả giải thích ngoài bài viết."
            + " Tôn trọng mọi ràng buộc của người dùng trong prompt (ví dụ cụm từ bắt buộc)."
            + " Nếu thiếu dữ liệu cho một ý bắt buộc, ghi đúng câu: "
            + f"\"{MISSING_INTERNAL_DATA_TEXT}\"."
        )

        try:
            response = self._llm.invoke(llm_instruction)
        except Exception as exc:
            self._record_llm_failure(llm_trace, "markdown_direct", str(exc))
            self._log_raw_llm_exchange("markdown_direct_error", llm_instruction, f"ERROR: {exc}")
            return None

        raw = str(getattr(response, "content", "") or "").strip()
        self._log_raw_llm_exchange("markdown_direct", llm_instruction, raw)
        if not raw:
            self._record_llm_failure(llm_trace, "markdown_direct", "empty_response")
            return None

        # Basic markdown shape check.
        if "##" not in raw and len(raw.split()) < 120:
            self._record_llm_failure(llm_trace, "markdown_direct", "insufficient_markdown_structure")
            return None

        try:
            return self._generated_from_markdown(raw, parsed, docs)
        except Exception:
            self._record_llm_failure(llm_trace, "markdown_direct", "markdown_parse_failure")
            return None

    def _generate_with_structured_output(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
        previous_draft: str | None,
        llm_trace: dict[str, Any] | None = None,
    ) -> GeneratedBlog | None:
        if self._llm is None or not settings.use_structured_output:
            return None
        if not hasattr(self._llm, "with_structured_output"):
            return None

        if isinstance(llm_trace, dict):
            llm_trace["attempted"] = True

        prompt_text = self._prompt_template.format(
            topic=parsed.topic,
            intent=parsed.intent,
            audience=parsed.audience,
            tone=parsed.tone,
            length=parsed.length,
            context=self._format_context(docs),
            custom_instructions=parsed.custom_instructions,
        )

        previous_note = ""
        if previous_draft:
            previous_note = f"\n\nPrevious draft context (truncated):\n{previous_draft[:500]}"

        instruction = (
            prompt_text
            + previous_note
            + "\n\nReturn structured data for an SEO blog with this schema: "
            + "title, introduction, sections[{header, content, image_search_keyword}], conclusion, meta_tags. "
            + "Use Vietnamese unless the user explicitly requests English. "
            + "Follow Chain-of-Verification internally before final output: verify every claim against retrieved context. "
            + f"If a required fact is missing, include exactly: \"{MISSING_INTERNAL_DATA_TEXT}\" "
            + "Append citations in this format: [Nguồn: Tiêu đề tài liệu | URL: đường_dẫn_nếu_có].\n"
            + "IMAGE REMOVAL: If the user asks to remove/delete an image from a specific section, "
            + "set image_search_keyword to exactly \"REMOVE_IMAGE\" for that section. "
            + "Keep all other sections' image_search_keyword as normal descriptive keywords."
        )

        try:
            structured_llm = self._llm.with_structured_output(BlogStructuredOutput)
            result = structured_llm.invoke(instruction)
        except Exception as exc:
            self._record_llm_failure(llm_trace, "structured_output", str(exc))
            self._log_raw_llm_exchange("structured_output_error", instruction, f"ERROR: {exc}")
            return None

        self._log_raw_llm_exchange("structured_output", instruction, str(result))

        try:
            if isinstance(result, BlogStructuredOutput):
                payload = result
            elif isinstance(result, dict):
                payload = BlogStructuredOutput.model_validate(result)
            elif hasattr(result, "model_dump"):
                payload = BlogStructuredOutput.model_validate(result.model_dump())
            else:
                self._record_llm_failure(llm_trace, "structured_output", "unexpected_structured_payload_type")
                return None
        except ValidationError:
            self._record_llm_failure(llm_trace, "structured_output", "schema_validation_error")
            return None

        sections: list[GeneratedBlog.Section] = []
        for item in payload.sections:
            heading = item.header.strip()
            body = item.content.strip()
            keyword = item.image_search_keyword.strip() or heading
            if not heading or not body:
                continue
            image_url, image_alt = self._resolve_section_image(parsed.topic, keyword, heading)
            sections.append(
                GeneratedBlog.Section(
                    heading=heading,
                    body=body,
                    image_url=image_url,
                    image_alt=image_alt,
                )
            )

        if not sections:
            self._record_llm_failure(llm_trace, "structured_output", "no_valid_sections")
            return None

        draft = self._render_structured_markdown(
            payload.title,
            payload.introduction,
            sections,
            payload.conclusion,
            payload.meta_tags,
        )
        sources_used = [str(doc.metadata.get("doc_id", "unknown_doc")) for doc in docs]

        return GeneratedBlog(
            title=payload.title,
            outline=[section.heading for section in sections],
            draft=draft,
            sources_used=sources_used,
            sections=sections,
        )

    def _generate_with_llm(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
        previous_draft: str | None,
        llm_trace: dict[str, Any] | None = None,
    ) -> GeneratedBlog | None:
        if self._llm is None:
            return None

        # Bypass structured output if the user demands custom formatting
        bypass_structured = not settings.use_structured_output or len(parsed.custom_instructions) > 2

        if bypass_structured:
            direct_markdown_result = self._generate_markdown_directly_with_llm(parsed, docs, previous_draft, llm_trace=llm_trace)
            if direct_markdown_result is not None:
                return direct_markdown_result

        structured_result = self._generate_with_structured_output(parsed, docs, previous_draft, llm_trace=llm_trace)
        if structured_result is not None:
            return structured_result

        if isinstance(llm_trace, dict):
            llm_trace["attempted"] = True

        prompt_text = self._prompt_template.format(
            topic=parsed.topic,
            intent=parsed.intent,
            audience=parsed.audience,
            tone=parsed.tone,
            length=parsed.length,
            context=self._format_context(docs),
            custom_instructions=parsed.custom_instructions,
        )

        follow_up_note = ""
        if previous_draft:
            follow_up_note = (
                "\n\nPrevious draft context (truncated):\n"
                f"{previous_draft[:500]}"
            )

        llm_instruction = (
            prompt_text
            + follow_up_note
            + "\n\nReturn valid JSON with keys: title, introduction, sections, conclusion, meta_tags, and optional draft/outline."
            + " Each section must include header, content, image_search_keyword."
            + " Keep content grounded in retrieved context and use Vietnamese unless user requests English."
            + " Apply Chain-of-Verification internally to check each factual claim before responding."
            + f" If evidence is missing, write exactly: \"{MISSING_INTERNAL_DATA_TEXT}\""
            + " Include citations in this format: [Nguồn: Tiêu đề tài liệu | URL: đường_dẫn_nếu_có]."
        )

        try:
            response = self._llm.invoke(llm_instruction)
        except Exception as exc:
            self._record_llm_failure(llm_trace, "json_output", str(exc))
            self._log_raw_llm_exchange("json_output_error", llm_instruction, f"ERROR: {exc}")
            return None

        raw = getattr(response, "content", "")
        if not isinstance(raw, str):
            self._record_llm_failure(llm_trace, "json_output", "non_string_response")
            return None

        self._log_raw_llm_exchange("json_output", llm_instruction, raw)

        parsed_json = self._extract_json_block(raw)
        if not parsed_json:
            self._record_llm_failure(llm_trace, "json_output", "invalid_json_response")
            return None

        title = str(parsed_json.get("title", "")).strip()
        outline = parsed_json.get("outline", [])
        introduction = str(parsed_json.get("introduction", "")).strip()
        conclusion = str(parsed_json.get("conclusion", "")).strip()
        meta_tags = str(parsed_json.get("meta_tags", "")).strip()
        draft = str(parsed_json.get("draft", "")).strip()
        sections_raw = parsed_json.get("sections", [])

        if not title:
            self._record_llm_failure(llm_trace, "json_output", "missing_title")
            return None
        if not isinstance(outline, list):
            outline = []

        retrieved_docs = self._docs_to_retrieved(docs)
        sections: list[GeneratedBlog.Section] = []
        if isinstance(sections_raw, list):
            for item in sections_raw:
                if not isinstance(item, dict):
                    continue
                heading = str(item.get("header") or item.get("heading") or "").strip()
                body = str(item.get("content") or item.get("body") or "").strip()
                image_keyword = str(item.get("image_search_keyword") or item.get("image_hint") or heading or parsed.topic).strip()
                if not heading or not body:
                    continue
                image_url, image_alt = self._resolve_section_image(parsed.topic, image_keyword, heading)
                sections.append(
                    GeneratedBlog.Section(
                        heading=heading,
                        body=body,
                        image_url=image_url,
                        image_alt=image_alt,
                    )
                )
        if not sections:
            sections = build_sections(parsed, [str(item) for item in outline], retrieved_docs)
        if not draft:
            draft = self._render_structured_markdown(
                title,
                introduction,
                sections,
                conclusion,
                meta_tags,
            )
        elif not draft.startswith("# "):
            draft = render_markdown_blog(title, sections)

        final_outline = [str(item).strip() for item in outline if str(item).strip()]
        if not final_outline:
            final_outline = [section.heading for section in sections]

        return GeneratedBlog(
            title=title,
            outline=final_outline,
            draft=draft,
            sources_used=[str(doc.metadata.get("doc_id", "unknown_doc")) for doc in docs],
            sections=sections,
        )

    def _generate_with_agent(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
        previous_draft: str | None,
    ) -> GeneratedBlog | None:
        if self._agent_executor is None:
            return None

        base_context = self._format_context(docs)
        previous_note = f"\nPrevious draft (truncated): {previous_draft[:400]}" if previous_draft else ""
        instruction = (
            "You are an Agentic RAG blog assistant.\n"
            "Task: produce a polished SEO blog for validex.com.au with section-level image keywords.\n"
            "You should reason step-by-step and call tools when useful:\n"
            "- Use database_vector_search or pinecone_search for factual grounding.\n"
            "- Use pinecone_search for legal/compliance facts.\n"
            "- Use validex_website_reader for Validex service/pricing context.\n"
            "- Use universal_web_scraper if you need to fetch real-time data from a specific external URL.\n"
            "- Use image_ocr_extractor if the user provides an image URL or you need to read an image.\n"
            "- Use unsplash_image_search to validate image keyword quality.\n"
            "- Use seo_blog_check on your draft before finalizing.\n"
            "Grounding and citation requirements:\n"
            "- Follow Chain-of-Verification internally: draft claims then verify each claim with retrieved/tool evidence.\n"
            "- Do not output unsupported facts.\n"
            + f"- If a requested fact is not present in evidence, use exactly: \"{MISSING_INTERNAL_DATA_TEXT}\"\n"
            "- Citations must follow: [Nguồn: Tiêu đề tài liệu | URL: đường_dẫn_nếu_có].\n"
            "Return ONLY JSON with keys: title, introduction, sections, conclusion, meta_tags.\n"
            "sections must be an array of objects: header, content, image_search_keyword.\n"
            f"Topic: {parsed.topic}\nAudience: {parsed.audience}\nTone: {parsed.tone}\nLength: {parsed.length}\n"
            f"Retrieved context:\n{base_context}{previous_note}"
        )

        try:
            if hasattr(self._agent_executor, "invoke"):
                result = self._agent_executor.invoke({"input": instruction})
                raw = result.get("output", "") if isinstance(result, dict) else str(result)
            else:
                raw = str(self._agent_executor.run(instruction))
        except Exception:
            return None

        parsed_json = self._extract_json_block(raw)
        if not parsed_json:
            return None

        title = str(parsed_json.get("title", "")).strip()
        outline_raw = parsed_json.get("outline", [])
        introduction = str(parsed_json.get("introduction", "")).strip()
        conclusion = str(parsed_json.get("conclusion", "")).strip()
        meta_tags = str(parsed_json.get("meta_tags", "")).strip()
        sections_raw = parsed_json.get("sections", [])
        if not title:
            return None

        outline = [str(item).strip() for item in outline_raw] if isinstance(outline_raw, list) else []
        sections: list[GeneratedBlog.Section] = []
        if isinstance(sections_raw, list):
            for item in sections_raw:
                if not isinstance(item, dict):
                    continue
                heading = str(item.get("header") or item.get("heading") or "").strip()
                body = str(item.get("content") or item.get("body") or "").strip()
                image_keyword = str(item.get("image_search_keyword") or item.get("image_hint") or heading or parsed.topic).strip()
                if heading and body:
                    image_url, image_alt = self._resolve_section_image(parsed.topic, image_keyword, heading)
                    sections.append(
                        GeneratedBlog.Section(
                            heading=heading,
                            body=body,
                            image_url=image_url,
                            image_alt=image_alt,
                        )
                    )

        if not sections:
            sections = build_sections(parsed, outline, self._docs_to_retrieved(docs))
        draft = self._render_structured_markdown(title, introduction, sections, conclusion, meta_tags)
        if not introduction and not conclusion:
            draft = render_markdown_blog(title, sections)

        final_outline = outline or [section.heading for section in sections]

        return GeneratedBlog(
            title=title,
            outline=final_outline,
            draft=draft,
            sources_used=[str(doc.metadata.get("doc_id", "unknown_doc")) for doc in docs],
            sections=sections,
        )

    def _generate(self, payload: dict) -> GeneratedBlog:
        parsed: ParsedPrompt = payload["effective_parsed"]
        previous_draft: str | None = payload.get("previous_draft")
        docs: list[Document] = payload["documents"]
        llm_trace = payload.get("llm_trace")
        conversation_history = payload.get("conversation_history")

        # Circuit breaker: skip LLM entirely when tripped
        if not self._cb_is_open():
            agent_result = self._generate_with_agent(parsed, docs, previous_draft)
            if agent_result is not None:
                self._last_generation_mode = "agentic"
                self._cb_record_success()
                logger.info("pipeline.generate_mode", extra={"mode": "agentic"})
                return self._enforce_grounding_and_citations(agent_result, docs)

            live_result = self._generate_with_llm(parsed, docs, previous_draft, llm_trace=llm_trace)
            if live_result is not None:
                self._last_generation_mode = "llm"
                self._cb_record_success()
                logger.info("pipeline.generate_mode", extra={"mode": "llm"})
                return self._enforce_grounding_and_citations(live_result, docs)

        self._last_generation_mode = "fallback"
        logger.info("pipeline.generate_mode", extra={"mode": "fallback"})
        fallback_result = self._generate_with_fallback(parsed, docs, previous_draft)
        return self._enforce_grounding_and_citations(fallback_result, docs)

    def run(
        self,
        prompt: str,
        *,
        previous_turn_topic: str | None = None,
        previous_draft: str | None = None,
        session: Any | None = None,
        request_id: str | None = None,
    ) -> dict:
        llm_trace: dict[str, Any] = {
            "attempted": False,
            "parse_mode": "heuristic",
            "failures": [],
            "request_id": request_id,
        }
        if request_id:
            logger.info("pipeline.run", extra={"request_id": request_id, "prompt_len": len(prompt)})
        # Build conversation history from session
        conversation_history = ""
        if session is not None and hasattr(session, "conversation_summary"):
            conversation_history = session.conversation_summary(max_turns=settings.max_conversation_turns)

        parsed = self._parse_chain.invoke({"prompt": prompt, "llm_trace": llm_trace})

        context_note = ""
        effective_parsed = parsed
        if parsed.intent in {"rewrite", "shorten"} and parsed.topic == "current draft" and previous_turn_topic:
            effective_parsed = replace(parsed, topic=previous_turn_topic)
            context_note = "context: using previous draft"

        token_plan = self._token_budget_plan(parsed.length)

        retrieval_bundle = self._retrieve_chain.invoke(
            {
                "effective_topic": effective_parsed.topic,
                "retrieval_top_k": token_plan.recommended_top_k,
            }
        )
        decision = retrieval_bundle.decision
        documents = retrieval_bundle.documents
        context_documents: list[Document] = []
        estimated_input_tokens = 0
        input_budget_sufficient = False
        quality_gate_blocked = False
        external_knowledge_used = False
        generation_mode = "fallback"
        if decision.reason == "pgvector retrieval successful":
            retrieval_mode = "pgvector"
        elif decision.reason == "vector retrieval successful":
            retrieval_mode = "pinecone"
        else:
            retrieval_mode = "local"

        if decision.status == "ok":
            context_documents, estimated_input_tokens = self._select_context_documents(documents, token_plan)
            input_budget_sufficient = estimated_input_tokens >= token_plan.input_tokens_min
            if not input_budget_sufficient:
                logger.warning(
                    "pipeline.context_budget_under_target",
                    extra={
                        "topic": effective_parsed.topic,
                        "estimated_input_tokens": estimated_input_tokens,
                        "target_min_tokens": token_plan.input_tokens_min,
                        "selected_docs": len(context_documents),
                        "retrieved_docs": len(documents),
                    },
                )
            # On-the-fly scraping: if user provided URLs, scrape them and append to context.
            scrape_urls = effective_parsed.modifiers.get("scrape_urls", [])
            for url in scrape_urls:
                scraped_text = self._tool_universal_web_scraper(url)
                if scraped_text and not scraped_text.startswith("Invalid URL") and not scraped_text.startswith("Scraping succeeded but no readable"):
                    external_knowledge_used = True
                    from langchain_core.documents import Document
                    synthetic_doc = Document(
                        page_content=scraped_text,
                        metadata={"source": "User Provided URL", "url": url, "doc_id": f"synthetic_url_{url}"}
                    )
                    context_documents.insert(0, synthetic_doc)
                    documents.insert(0, synthetic_doc)

            context_text = self._format_context(
                context_documents,
                snippet_chars=max(220, settings.chunk_token_estimate * 4),
            )
            self._prompt_template.format(
                topic=effective_parsed.topic,
                intent=parsed.intent,
                audience=parsed.audience,
                tone=parsed.tone,
                length=parsed.length,
                context=context_text,
                custom_instructions=effective_parsed.custom_instructions,
            )

            # Check cache before generating (skip for rewrite/shorten)
            cache_key = None
            _skip_cache_intents = {"rewrite", "shorten"}
            if effective_parsed.intent not in _skip_cache_intents:
                from app.cache import ResponseCache
                cache_key = ResponseCache.make_key(
                    effective_parsed.topic, effective_parsed.intent, effective_parsed.length,
                )
                cached = response_cache.get(cache_key)
                if cached is not None:
                    generation_mode = "cached"
                    generated_payload = cached
                    # Skip generation entirely — jump to return
                    quality_gate_blocked = False
                    estimated_output_tokens = self._estimate_token_count(str(generated_payload.get("draft", "")))
                    # (jump past the generation block below)
                    cache_key = None  # prevent re-caching

            if generation_mode == "fallback":  # not yet generated (no cache hit)
                generated = self._generate_chain.invoke(
                    {
                        "effective_parsed": effective_parsed,
                        "previous_draft": previous_draft,
                        "documents": context_documents,
                        "llm_trace": llm_trace,
                        "conversation_history": conversation_history,
                    }
                )
                generation_mode = self._last_generation_mode

                quality_ok, quality_reason = self._quality_gate_result(generated)
                if not quality_ok:
                    quality_gate_blocked = True
                    logger.warning(
                        "pipeline.quality_gate_blocked",
                        extra={
                            "topic": effective_parsed.topic,
                            "reason": quality_reason,
                            "sections": len(generated.sections),
                            "sources": len(generated.sources_used),
                        },
                    )
                    generated_payload = {
                        "title": "Need More Context",
                        "outline": [
                            "Bo sung them nguon tai lieu lien quan",
                            "Lam ro pham vi blog can tao",
                            "Thu lai prompt voi yeu cau cu the hon",
                        ],
                        "draft": (
                            "Ban nhap hien tai chua dat quality gate de xuat ban. "
                            f"Ly do: {quality_reason}. "
                            "Hay bo sung du lieu va thu lai prompt."
                        ),
                        "sources_used": generated.sources_used,
                        "sections": [],
                    }
                else:
                    generated_payload = self._generated_to_payload(generated)
                    generated_payload = self._apply_prompt_edit_constraints(parsed.raw_prompt, generated_payload)
                    # Store in cache for future identical queries
                    if cache_key:
                        response_cache.put(cache_key, generated_payload)
        else:
            if settings.allow_hybrid_fallback:
                external_knowledge_used = True
                generation_mode = "hybrid_fallback"
                hybrid_generated = self._generate_with_hybrid_fallback(
                    effective_parsed,
                    previous_draft,
                    decision.status,
                )
                grounded_hybrid = self._enforce_grounding_and_citations(hybrid_generated, [])
                generated_payload = self._generated_to_payload(grounded_hybrid)
                generated_payload = self._apply_prompt_edit_constraints(parsed.raw_prompt, generated_payload)
            else:
                generation_mode = "blocked"
                reason_map = {
                    "out_of_domain": "Query hien tai nam ngoai pham vi dataset RAG hien co.",
                    "low_confidence": "Do lien quan retrieval qua thap de tao draft an toan.",
                    "no_match": "Khong tim thay tai lieu phu hop trong knowledge base.",
                    "no_data": "He thong chua co du lieu processed de retrieval.",
                }
                generated_payload = {
                    "title": "Need More Context",
                    "outline": [
                        "Xac dinh lai chu de trong pham vi dataset",
                        "Bo sung tai lieu lien quan vao data/raw",
                        "Chay ingest de cap nhat data/processed va metadata",
                        "Gui lai prompt cu the hon",
                    ],
                    "draft": (
                        "Toi chua the tao draft dang tin cay cho prompt nay. "
                        f"Ly do: {reason_map.get(decision.status, decision.reason)} "
                        "Hay thu prompt cu the hon trong domain police check/recruitment/compliance, "
                        "hoac cap nhat dataset RAG truoc khi tao noi dung."
                    ),
                    "sources_used": [],
                    "sections": [],
                }

        estimated_output_tokens = self._estimate_token_count(str(generated_payload.get("draft", "")))
        llm_failures_raw = llm_trace.get("failures") if isinstance(llm_trace, dict) else []
        llm_failures = llm_failures_raw if isinstance(llm_failures_raw, list) else []
        fallback_reason: str | None = None
        if generation_mode == "fallback":
            if llm_failures:
                last_reason = str(llm_failures[-1].get("reason", "") or "").strip()
                fallback_reason = last_reason or "llm_error"
            elif self._llm is None or not settings.use_live_llm:
                fallback_reason = "llm_not_available"
            else:
                fallback_reason = "llm_no_valid_output"
        elif generation_mode == "hybrid_fallback":
            fallback_reason = f"retrieval_{decision.status}"
        elif generation_mode == "blocked":
            fallback_reason = f"retrieval_{decision.status}"

        return {
            "parsed": {
                "intent": parsed.intent,
                "topic": effective_parsed.topic,
                "audience": parsed.audience,
                "tone": parsed.tone,
                "length": parsed.length,
                "context_note": context_note,
            },
            "retrieved": [
                {
                    "doc_id": str(doc.metadata.get("doc_id", "unknown_doc")),
                    "score": int(doc.metadata.get("score", 0)),
                    "semantic_score": round(float(doc.metadata.get("semantic_score", 0.0)), 3),
                    "snippet": doc.page_content[:220],
                }
                for doc in documents
            ],
            "retrieval_meta": {
                "status": decision.status,
                "confidence": round(decision.confidence, 3),
                "top_score": decision.top_score,
                "reason": decision.reason,
            },
            "runtime": {
                "generation_mode": generation_mode,
                "retrieval_mode": retrieval_mode,
                "quality_gate_blocked": quality_gate_blocked,
                "external_knowledge_used": external_knowledge_used,
                "fallback_reason": fallback_reason,
                "llm_debug": {
                    "attempted": bool(llm_trace.get("attempted")),
                    "parse_mode": str(llm_trace.get("parse_mode", "heuristic")),
                    "failure_count": len(llm_failures),
                    "failures": llm_failures[-3:],
                },
                "token_budget": {
                    "length_profile": token_plan.length_profile,
                    "output_tokens_target": token_plan.output_tokens,
                    "output_tokens_estimated": estimated_output_tokens,
                    "input_tokens_target_min": token_plan.input_tokens_min,
                    "input_tokens_target": token_plan.input_tokens_target,
                    "input_tokens_target_max": token_plan.input_tokens_max,
                    "input_tokens_estimated": estimated_input_tokens,
                    "input_budget_sufficient": input_budget_sufficient,
                    "recommended_top_k": token_plan.recommended_top_k,
                    "retrieved_docs": len(documents),
                    "context_docs_used": len(context_documents),
                },
            },
            "generated": generated_payload,
        }

    @staticmethod
    def _format_context(documents: list[Document], snippet_chars: int = 220) -> str:
        if not documents:
            return "Content: (không có dữ liệu)\nMetadata: title=không có; url=không có; region=không rõ"
        lines: list[str] = []
        max_chars = max(120, snippet_chars)
        for index, doc in enumerate(documents, start=1):
            doc_id = doc.metadata.get("doc_id", "unknown_doc")
            title = str(doc.metadata.get("title", "")).strip() or "untitled"
            source_url = str(doc.metadata.get("source_url", "")).strip() or "n/a"
            region = str(doc.metadata.get("region", "")).strip() or "AU"
            page_ref = str(doc.metadata.get("page") or doc.metadata.get("page_number") or doc.metadata.get("chunk_id") or "n/a")
            snippet = re.sub(r"\s+", " ", doc.page_content).strip()[:max_chars]
            lines.append(f"[Tài liệu {index}]")
            lines.append(f"Content: {snippet}")
            lines.append(
                f"Metadata: title={title}; url={source_url}; region={region}; doc_id={doc_id}; page={page_ref}"
            )
            lines.append("")
        return "\n".join(lines).strip()


pipeline = LangChainRAGPipeline()
