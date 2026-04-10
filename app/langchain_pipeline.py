from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
import importlib
import logging
import re
from urllib.parse import urlparse
from urllib.request import urlopen
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from app.config import settings
from app.generator import build_section_image_url
from app.generator import build_sections
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


logger = logging.getLogger(__name__)

@dataclass
class RetrievalBundle:
    decision: RetrievalDecision
    documents: list[Document]


class LangChainRAGPipeline:
    """LangChain-based orchestration for parse -> retrieve -> generate."""

    def __init__(self) -> None:
        self._llm = self._build_llm()
        self._vector_store = self._build_vector_store()
        self._agent_executor = self._build_agent_executor()
        self._parse_chain = RunnableLambda(self._parse)
        self._retrieve_chain = RunnableLambda(self._retrieve)
        self._prompt_template = PromptTemplate.from_template(
            """
            You are an assistant preparing a grounded blog draft.
            Topic: {topic}
            Intent: {intent}
            Audience: {audience}
            Tone: {tone}
            Length: {length}

            Retrieved context:
            {context}
            """.strip()
        )
        self._generate_chain = RunnableLambda(self._generate)

    def runtime_status(self) -> dict[str, Any]:
        if self._vector_store is not None and settings.use_pinecone_retrieval:
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
                name="seo_blog_check",
                func=self._tool_seo_blog_check,
                description=(
                    "Validate SEO quality of a blog draft. Input must be the current draft text."
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

    def _tool_pinecone_search(self, query: str) -> str:
        bundle = self._retrieve_from_pinecone(query)
        if not bundle.documents:
            return "No relevant vector results found."
        lines = []
        for doc in bundle.documents[:3]:
            lines.append(
                f"[{doc.metadata.get('doc_id', 'unknown_doc')}] "
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

        try:
            with urlopen(settings.validex_website_url, timeout=8) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return "Unable to fetch website content right now."

        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "Website content is empty after cleanup."
        return text[:2200]

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

    def _parse(self, payload: dict) -> ParsedPrompt:
        return parse_user_input(payload["prompt"])

    def _build_llm(self) -> Any | None:
        if not settings.use_live_llm or not settings.openai_api_key:
            return None
        if ChatOpenAI is None:
            return None
        try:
            return ChatOpenAI(
                model=settings.model_name,
                api_key=settings.openai_api_key,
                temperature=0.2,
            )
        except Exception:
            return None

    def _build_vector_store(self) -> Any | None:
        if not settings.use_pinecone_retrieval:
            return None
        if not (settings.pinecone_api_key and settings.pinecone_index and settings.openai_api_key):
            return None
        if OpenAIEmbeddings is None:
            return None

        try:
            pinecone_module = importlib.import_module("langchain_pinecone")
            pinecone_vector_store_cls = getattr(pinecone_module, "PineconeVectorStore", None)
        except Exception:
            pinecone_vector_store_cls = None

        if pinecone_vector_store_cls is None:
            return None

        try:
            embeddings = OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=settings.openai_api_key,
            )
            return pinecone_vector_store_cls(
                index_name=settings.pinecone_index,
                embedding=embeddings,
                namespace=settings.pinecone_namespace or None,
                pinecone_api_key=settings.pinecone_api_key,
                text_key="text",
            )
        except TypeError:
            try:
                return pinecone_vector_store_cls.from_existing_index(
                    index_name=settings.pinecone_index,
                    embedding=embeddings,
                    namespace=settings.pinecone_namespace or None,
                    text_key="text",
                    pinecone_api_key=settings.pinecone_api_key,
                )
            except Exception:
                return None
        except Exception:
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
    def _get_doc_id(doc: Document, fallback_id: str) -> str:
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("id")
        if not doc_id:
            source = str(doc.metadata.get("source", ""))
            stem = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            doc_id = stem.replace(".txt", "") if stem else fallback_id
        return str(doc_id)

    def _retrieve_from_pinecone(self, query: str) -> RetrievalBundle:
        if self._vector_store is None:
            return self._retrieve_from_local_guard(query)

        docs_with_scores: list[tuple[Document, float]] = []
        try:
            if hasattr(self._vector_store, "similarity_search_with_relevance_scores"):
                raw_results = self._vector_store.similarity_search_with_relevance_scores(query, k=settings.top_k)
                docs_with_scores = [(doc, float(score)) for doc, score in raw_results]
            elif hasattr(self._vector_store, "similarity_search_with_score"):
                raw_results = self._vector_store.similarity_search_with_score(query, k=settings.top_k)
                docs_with_scores = [(doc, float(score)) for doc, score in raw_results]
            else:
                docs = self._vector_store.similarity_search(query, k=settings.top_k)
                docs_with_scores = [(doc, 0.6) for doc in docs]
        except Exception:
            return self._retrieve_from_local_guard(query)

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

    def _retrieve_from_local_guard(self, query: str) -> RetrievalBundle:
        decision = retrieve_with_guard(
            query,
            settings.data_processed_dir,
            settings.top_k,
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
        logger.info("pipeline.retrieve_start", extra={"topic": topic})
        if settings.use_pinecone_retrieval:
            bundle = self._retrieve_from_pinecone(topic)
        else:
            bundle = self._retrieve_from_local_guard(topic)
        logger.info(
            "pipeline.retrieve_done",
            extra={
                "topic": topic,
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

    def _generate_with_llm(
        self,
        parsed: ParsedPrompt,
        docs: list[Document],
        previous_draft: str | None,
    ) -> GeneratedBlog | None:
        if self._llm is None:
            return None

        prompt_text = self._prompt_template.format(
            topic=parsed.topic,
            intent=parsed.intent,
            audience=parsed.audience,
            tone=parsed.tone,
            length=parsed.length,
            context=self._format_context(docs),
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
            + "\n\nReturn a valid JSON object with keys: title (string), outline (array of strings), draft (string), sections (array)."
            + " Each item in sections must contain: heading, body, image_hint."
            + " Keep content grounded in retrieved context and avoid unsupported claims."
        )

        try:
            response = self._llm.invoke(llm_instruction)
        except Exception:
            return None

        raw = getattr(response, "content", "")
        if not isinstance(raw, str):
            return None

        parsed_json = self._extract_json_block(raw)
        if not parsed_json:
            return None

        title = str(parsed_json.get("title", "")).strip()
        outline = parsed_json.get("outline", [])
        draft = str(parsed_json.get("draft", "")).strip()
        sections_raw = parsed_json.get("sections", [])

        if not title or not draft:
            return None
        if not isinstance(outline, list):
            outline = []

        retrieved_docs = self._docs_to_retrieved(docs)
        sections: list[GeneratedBlog.Section] = []
        if isinstance(sections_raw, list):
            for item in sections_raw:
                if not isinstance(item, dict):
                    continue
                heading = str(item.get("heading", "")).strip()
                body = str(item.get("body", "")).strip()
                image_hint = str(item.get("image_hint", heading or parsed.topic)).strip()
                if not heading or not body:
                    continue
                sections.append(
                    GeneratedBlog.Section(
                        heading=heading,
                        body=body,
                        image_url=build_section_image_url(parsed.topic, image_hint),
                        image_alt=f"{heading} illustration",
                    )
                )
        if not sections:
            sections = build_sections(parsed, [str(item) for item in outline], retrieved_docs)
        if not draft.startswith("# "):
            draft = render_markdown_blog(title, sections)

        return GeneratedBlog(
            title=title,
            outline=[str(item).strip() for item in outline if str(item).strip()],
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
            "Task: produce a polished blog with sections and image ideas.\n"
            "You should reason step-by-step and call tools when useful:\n"
            "- Use pinecone_search for legal/compliance facts.\n"
            "- Use validex_website_reader for service/pricing context if available.\n"
            "- Use seo_blog_check on your draft before finalizing.\n"
            "Return ONLY JSON with keys: title, outline, sections.\n"
            "sections must be an array of objects: heading, body, image_hint.\n"
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
        sections_raw = parsed_json.get("sections", [])
        if not title:
            return None

        outline = [str(item).strip() for item in outline_raw] if isinstance(outline_raw, list) else []
        sections: list[GeneratedBlog.Section] = []
        if isinstance(sections_raw, list):
            for item in sections_raw:
                if not isinstance(item, dict):
                    continue
                heading = str(item.get("heading", "")).strip()
                body = str(item.get("body", "")).strip()
                image_hint = str(item.get("image_hint", heading or parsed.topic)).strip()
                if heading and body:
                    sections.append(
                        GeneratedBlog.Section(
                            heading=heading,
                            body=body,
                            image_url=build_section_image_url(parsed.topic, image_hint),
                            image_alt=f"{heading} illustration",
                        )
                    )

        if not sections:
            sections = build_sections(parsed, outline, self._docs_to_retrieved(docs))
        draft = render_markdown_blog(title, sections)

        return GeneratedBlog(
            title=title,
            outline=outline,
            draft=draft,
            sources_used=[str(doc.metadata.get("doc_id", "unknown_doc")) for doc in docs],
            sections=sections,
        )

    def _generate(self, payload: dict) -> GeneratedBlog:
        parsed: ParsedPrompt = payload["effective_parsed"]
        previous_draft: str | None = payload.get("previous_draft")
        docs: list[Document] = payload["documents"]

        agent_result = self._generate_with_agent(parsed, docs, previous_draft)
        if agent_result is not None:
            logger.info("pipeline.generate_mode", extra={"mode": "agentic"})
            return agent_result

        live_result = self._generate_with_llm(parsed, docs, previous_draft)
        if live_result is not None:
            logger.info("pipeline.generate_mode", extra={"mode": "llm"})
            return live_result
        logger.info("pipeline.generate_mode", extra={"mode": "fallback"})
        return self._generate_with_fallback(parsed, docs, previous_draft)

    def run(
        self,
        prompt: str,
        *,
        previous_turn_topic: str | None = None,
        previous_draft: str | None = None,
    ) -> dict:
        parsed = self._parse_chain.invoke({"prompt": prompt})

        context_note = ""
        effective_parsed = parsed
        if parsed.intent in {"rewrite", "shorten"} and parsed.topic == "current draft" and previous_turn_topic:
            effective_parsed = replace(parsed, topic=previous_turn_topic)
            context_note = "context: using previous draft"

        retrieval_bundle = self._retrieve_chain.invoke({"effective_topic": effective_parsed.topic})
        decision = retrieval_bundle.decision
        documents = retrieval_bundle.documents
        quality_gate_blocked = False
        generation_mode = self.runtime_status().get("generation_mode", "fallback")
        retrieval_mode = "pinecone" if decision.reason == "vector retrieval successful" else "local"

        if decision.status == "ok":
            context_text = self._format_context(documents)
            self._prompt_template.format(
                topic=effective_parsed.topic,
                intent=parsed.intent,
                audience=parsed.audience,
                tone=parsed.tone,
                length=parsed.length,
                context=context_text,
            )

            generated = self._generate_chain.invoke(
                {
                    "effective_parsed": effective_parsed,
                    "previous_draft": previous_draft,
                    "documents": documents,
                }
            )

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
                generated_payload = {
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
        else:
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
            },
            "generated": generated_payload,
        }

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        if not documents:
            return "- No retrieval context available"
        lines: list[str] = []
        for doc in documents[:3]:
            doc_id = doc.metadata.get("doc_id", "unknown_doc")
            lines.append(f"- [{doc_id}] {doc.page_content[:220]}")
        return "\n".join(lines)


pipeline = LangChainRAGPipeline()
