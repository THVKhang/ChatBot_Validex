"""Smart Researcher Node — multi-query retrieval, deep scraping, LLM summarization."""
import json
import logging
import re
from langchain_core.documents import Document
from app.graph_state import GraphState, RetrievedDoc
from app.langchain_pipeline import pipeline
from app.config import settings

logger = logging.getLogger(__name__)

# ── Research Cache (in-memory, resets on server restart) ──────────
_research_cache: dict[str, list[RetrievedDoc]] = {}
_research_embeddings: dict[str, list[float]] = {}
_RESEARCH_CACHE_THRESHOLD = 0.85
_RESEARCH_CACHE_MAX = 50  # Prevent unbounded memory growth


def _check_research_cache(topic: str) -> list[RetrievedDoc] | None:
    """Check if a semantically similar topic was already researched."""
    if not _research_cache:
        return None
    try:
        from app.ingest_pgvector import get_embeddings_model
        model, _ = get_embeddings_model()
        if not model:
            return None
        query_emb = model.embed_query(topic)
        best_sim, best_topic = 0.0, None
        for cached_topic, cached_emb in _research_embeddings.items():
            # Cosine similarity (vectors are typically normalized)
            sim = sum(a * b for a, b in zip(query_emb, cached_emb))
            if sim > best_sim:
                best_sim, best_topic = sim, cached_topic
        if best_sim >= _RESEARCH_CACHE_THRESHOLD and best_topic:
            logger.info("Research Cache HIT: '%s' ≈ '%s' (sim=%.3f)", topic[:40], best_topic[:40], best_sim)
            return _research_cache[best_topic]
    except Exception as exc:
        logger.debug("Research cache check failed: %s", exc)
    return None


def _save_research_cache(topic: str, docs: list[RetrievedDoc]) -> None:
    """Save research results to in-memory cache."""
    if len(_research_cache) >= _RESEARCH_CACHE_MAX:
        # Evict oldest entry
        oldest = next(iter(_research_cache))
        _research_cache.pop(oldest, None)
        _research_embeddings.pop(oldest, None)
    try:
        from app.ingest_pgvector import get_embeddings_model
        model, _ = get_embeddings_model()
        if model:
            _research_embeddings[topic] = model.embed_query(topic)
            _research_cache[topic] = docs
            logger.info("Research Cache SAVED: '%s' (%d docs)", topic[:40], len(docs))
    except Exception as exc:
        logger.debug("Research cache save failed: %s", exc)


def _compress_doc_content(doc: Document, max_chars: int = 800) -> Document:
    """Remove redundant sentences within a single document."""
    sentences = doc.page_content.split('. ')
    seen: set[str] = set()
    unique: list[str] = []
    for s in sentences:
        key = ' '.join(s.lower().split()[:6])
        if key and key not in seen:
            seen.add(key)
            unique.append(s)
    doc.page_content = '. '.join(unique)[:max_chars]
    return doc

# Broad set of search concepts to compare the user's topic against
_PREDEFINED_SEARCH_CONCEPTS = [
    "Australian National Police Check",
    "criminal history check online",
    "working with children check (WWCC) requirements",
    "NDIS worker screening check",
    "Aged care worker compliance screening",
    "background check for employment in Australia",
    "Australian privacy law spent convictions",
    "Fair Work Act workplace screening",
    "right to work visa verification",
    "how long does a police check last validity",
    "police check processing time and cost",
    "state police clearance NSW VIC QLD WA SA",
    "what shows up on a police check result",
    "volunteer background screening requirements",
    "AFP national police certificate application",
]

def _expand_query(topic: str) -> list[str]:
    """Generate diverse search queries using Local Semantics (0 LLM tokens)."""
    try:
        from app.local_semantics import get_embedding, get_embeddings, batch_cosine_similarity
        
        topic_emb = get_embedding(topic)
        concept_embs = get_embeddings(_PREDEFINED_SEARCH_CONCEPTS)
        
        sims = batch_cosine_similarity(topic_emb, concept_embs)
        
        # Get indices of top 3 most similar concepts
        # Convert to float to avoid numpy types, ensure at least some similarity
        top_indices = [int(i) for i in sims.argsort()[::-1][:3] if sims[i] > 0.3]
        
        queries = [topic]
        if "australia" not in topic.lower() and "au" not in topic.lower():
            queries.append(f"{topic} Australia")
            
        for idx in top_indices:
            concept = _PREDEFINED_SEARCH_CONCEPTS[idx]
            if concept.lower() not in [q.lower() for q in queries]:
                queries.append(concept)
                
        # Limit to 4 queries total
        logger.info("Query expansion (Semantic): %s -> %s", topic[:40], queries[:4])
        return queries[:4]
        
    except Exception as exc:
        logger.error(f"Semantic query expansion failed: {exc}. Falling back to topic.")
        return [topic, f"{topic} Australia"]


def _extractive_summarize(text: str, topic: str, max_output: int = 400) -> str:
    """Extract most relevant sentences using Local Semantic Similarity (0 LLM tokens)."""
    if not text:
        return ""
    
    try:
        from app.local_semantics import get_embedding, get_embeddings, batch_cosine_similarity
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text[:3000]) if len(s.split()) >= 5]
        if not sentences:
            return text[:max_output]
            
        topic_emb = get_embedding(topic)
        sentence_embs = get_embeddings(sentences)
        
        sims = batch_cosine_similarity(topic_emb, sentence_embs)
        
        # Sort by relevance
        scored = list(zip(sims, sentences))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        result = []
        char_count = 0
        for sim, sentence in scored:
            # Only include sentences that are at least somewhat relevant
            if sim < 0.15:
                continue
            if char_count + len(sentence) > max_output:
                break
            result.append(sentence)
            char_count += len(sentence) + 1
            
        # Re-sort sentences chronologically based on their original order
        # to maintain some coherence (optional, but good practice for summaries)
        final_sentences = [s for s in sentences if s in result]
        return ' '.join(final_sentences) if final_sentences else text[:max_output]
        
    except Exception as exc:
        logger.error(f"Semantic extractive summarization failed: {exc}")
        return text[:max_output]


def _score_doc_relevance(doc_text: str, topic: str) -> float:
    """Quick heuristic relevance score (0-1) without calling LLM.
    
    Includes topic isolation: penalizes docs about different check types
    to prevent WWCC contamination in police check articles and vice versa.
    """
    topic_words = set(re.findall(r"\w+", topic.lower()))
    doc_words = set(re.findall(r"\w+", doc_text[:500].lower()))
    if not topic_words:
        return 0.5
    overlap = len(topic_words & doc_words)
    base_score = min(1.0, overlap / max(1, len(topic_words)))
    
    # ── Topic Isolation Penalty ──
    # Prevent cross-contamination between different check types
    topic_lower = topic.lower()
    doc_lower = doc_text[:1000].lower()
    
    # Define topic boundaries
    is_police_check_query = any(t in topic_lower for t in ["police check", "criminal history", "national police"])
    is_wwcc_query = any(t in topic_lower for t in ["working with children", "wwcc", "child-related"])
    is_ndis_query = any(t in topic_lower for t in ["ndis", "disability"])
    
    # Penalize off-topic documents
    if is_police_check_query and not is_wwcc_query:
        wwcc_signals = sum(1 for s in ["working with children", "wwcc", "child-related work", "worker screening act"]
                          if s in doc_lower)
        if wwcc_signals >= 2:
            base_score *= 0.4  # Heavy penalty for WWCC-heavy docs
            
    if is_wwcc_query and not is_police_check_query:
        police_signals = sum(1 for s in ["criminal history check", "nationally coordinated", "acic"]
                           if s in doc_lower)
        if police_signals >= 2 and "children" not in doc_lower:
            base_score *= 0.4
            
    if is_ndis_query:
        if "children" in doc_lower and "ndis" not in doc_lower:
            base_score *= 0.5
    
    return base_score


def _deduplicate_docs(docs: list[Document], threshold: float = 0.8) -> list[Document]:
    """Remove near-duplicate documents by comparing first 200 chars."""
    seen_fingerprints: list[str] = []
    unique = []
    for doc in docs:
        fp = doc.page_content[:200].lower().strip()
        is_dup = False
        for existing in seen_fingerprints:
            # Simple word overlap check
            fp_words = set(fp.split())
            ex_words = set(existing.split())
            if fp_words and ex_words:
                overlap = len(fp_words & ex_words) / max(len(fp_words), len(ex_words))
                if overlap > threshold:
                    is_dup = True
                    break
        if not is_dup:
            seen_fingerprints.append(fp)
            unique.append(doc)
    return unique


def _web_search_with_scraping(topic: str, max_results: int = 3) -> list[Document]:
    """Search DuckDuckGo + optionally scrape top URLs for richer content."""
    docs = []

    # DuckDuckGo search
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(topic, max_results=max_results + 2))
    except Exception as exc:
        logger.warning(f"DuckDuckGo search failed: {exc}")
        ddg_results = []

    # Google Custom Search fallback
    if not ddg_results and settings.google_search_api_key:
        try:
            from app.agents.discovery_agent import _google_custom_search
            ddg_results = _google_custom_search(topic, num=max_results)
            # Normalize keys
            ddg_results = [
                {"title": r.get("title", ""), "href": r.get("url", ""), "body": r.get("snippet", "")}
                for r in ddg_results
            ]
        except Exception as exc:
            logger.warning(f"Google search fallback failed: {exc}")

    if not ddg_results:
        return docs

    # Deep scrape top URLs for richer context
    urls_to_scrape = [r.get("href", "") for r in ddg_results if r.get("href")]
    scraped_content: dict[str, str] = {}

    try:
        from app.agents.scraper import scrape_multiple
        scraped = scrape_multiple(urls_to_scrape, max_urls=2, timeout=8)
        for item in scraped:
            # Summarize long content
            summary = _extractive_summarize(item["text"], topic)
            scraped_content[item["url"]] = summary
    except Exception as exc:
        logger.warning(f"Deep scraping failed: {exc}")

    for i, r in enumerate(ddg_results[:max_results]):
        url = r.get("href", "")
        title = r.get("title", "")
        snippet = r.get("body", "")

        # Use scraped + summarized content if available, otherwise use snippet
        content = scraped_content.get(url, snippet)
        if not content:
            content = snippet

        docs.append(Document(
            page_content=content,
            metadata={
                "doc_id": f"web_{i}",
                "title": title,
                "source_url": url,
                "source": "Web Search",
                "score": 100,
                "semantic_score": 1.0,
            }
        ))

    return docs


from app.agents.base import BaseAgentNode

class ResearcherAgentNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Smart researcher: multi-query retrieval + web search + deep scraping."""
        logger.info("Executing Smart Researcher Node")
        
        parsed = state["parsed"]
        topic = parsed.get("topic", "")
        session = state["session"]
    
        # Check for uploaded document
        last_turn = session.latest_turn()
        uploaded_text = getattr(last_turn, "uploaded_file_content", None) if last_turn else None
        
        if uploaded_text:
            docs = [RetrievedDoc(
                doc_id="uploaded_file", content=uploaded_text,
                score=100.0, source="User Upload", title="", source_url=""
            )]
            return {
                "retrieved_docs": docs,
                "retrieval_meta": {
                    "status": "ok",
                    "confidence": 1.0,
                    "top_score": 100,
                    "reason": "user uploaded file"
                }
            }
        
        # Check Agentic Loop State
        attempts = state.get("retrieval_attempts", 0)
        tried_queries = state.get("tried_queries", [])
        rag_feedback = state.get("rag_feedback")
        
        if rag_feedback:
            logger.warning(f"Researcher Retrying (Attempt {attempts + 1}). Previous Feedback: {rag_feedback}")
        
        # --- Research Cache: skip LLM if similar topic already researched ---
        # Only use cache on first attempt to avoid infinite loops if cache is bad
        if attempts == 0:
            cached_docs = _check_research_cache(topic)
            if cached_docs:
                logger.info("Researcher: returning %d cached docs (0 LLM tokens)", len(cached_docs))
                return {
                    "retrieved_docs": cached_docs,
                    "retrieval_attempts": attempts + 1
                }
            
        # --- Multi-query retrieval ---
        from app.langchain_pipeline import pipeline
        parsed_length = parsed.get("length", "medium")
        token_plan = pipeline._token_budget_plan(parsed_length)
        recommended_top_k = token_plan.recommended_top_k
    
        expanded_queries = _expand_query(topic)
        all_documents: list[Document] = []
        
        complexity_level = state.get("complexity_level", "simple")
        first_decision = None
        for i, query in enumerate(expanded_queries):
            payload = {
                "effective_topic": query,
                "retrieval_top_k": recommended_top_k,
                "complexity_level": complexity_level
            }
            try:
                bundle = pipeline._retrieve(payload)
                if i == 0:
                    first_decision = bundle.decision
                all_documents.extend(bundle.documents)
            except Exception as exc:
                logger.warning(f"Retrieval failed for query '{query}': {exc}")
        
        # Deduplicate
        all_documents = _deduplicate_docs(all_documents)
        
        # Score relevance and sort
        for doc in all_documents:
            rel_score = _score_doc_relevance(doc.page_content, topic)
            doc.metadata["relevance_score"] = rel_score
        
        all_documents.sort(key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        
        # If not enough good results, do web search with deep scraping
        good_docs = [d for d in all_documents if d.metadata.get("relevance_score", 0) > 0.3]
        
        if len(good_docs) < 2 and settings.allow_hybrid_fallback:
            logger.info(f"Researcher: only {len(good_docs)} good docs, triggering web search + scraping")
            web_docs = _web_search_with_scraping(topic, max_results=3)
            all_documents = good_docs + web_docs
        else:
            all_documents = good_docs
    
        # Limit to top recommended_top_k
        all_documents = all_documents[:recommended_top_k]
        
        # ── Reranker: Cross-Encoder reranking for precise relevance ordering ──
        # Uses ms-marco-MiniLM-L-12-v2 (ONNX, 0 API tokens, ~50ms latency)
        try:
            from app.reranker import get_reranker
            reranker = get_reranker()
            if all_documents:
                # Convert Documents to dicts for reranker
                doc_dicts = []
                for d in all_documents:
                    doc_dicts.append({
                        "doc_id": d.metadata.get("doc_id", ""),
                        "content": d.page_content,
                        "score": d.metadata.get("score", 0),
                        "source": d.metadata.get("source", ""),
                        "title": d.metadata.get("title", ""),
                        "source_url": d.metadata.get("source_url", ""),
                        "relevance_score": d.metadata.get("relevance_score", 0),
                    })
                reranked = reranker.rerank(topic, doc_dicts, top_k=recommended_top_k)
                # Convert back to Document objects
                reranked_docs = []
                for rd in reranked:
                    reranked_docs.append(Document(
                        page_content=rd["content"],
                        metadata={k: v for k, v in rd.items() if k != "content"},
                    ))
                all_documents = reranked_docs
                logger.info("Researcher: reranked %d documents by Cross-Encoder", len(all_documents))
        except Exception as exc:
            logger.warning("Reranker unavailable, keeping heuristic order: %s", exc)
        
        # Sentence-level deduplication within each doc
        all_documents = [_compress_doc_content(d) for d in all_documents]
        
        # Convert to RetrievedDoc
        docs = []
        for d in all_documents:
            docs.append(RetrievedDoc(
                doc_id=d.metadata.get("doc_id", "unknown"),
                content=d.page_content,
                score=d.metadata.get("score", 0.0),
                source=d.metadata.get("source", "Internal Database"),
                title=d.metadata.get("title", ""),
                source_url=d.metadata.get("source_url", ""),
            ))
    
        # Save to research cache for future similar topics (only if first attempt)
        if attempts == 0:
            _save_research_cache(topic, docs)
        
        # Update tried queries
        new_tried = list(set(tried_queries + expanded_queries))
        
        # Construct retrieval_meta from first_decision if available, else default
        ret_meta = {
            "status": "ok",
            "confidence": 1.0,
            "top_score": 100,
            "reason": "default decision status"
        }
        if first_decision:
            ret_meta = {
                "status": getattr(first_decision, "status", "ok"),
                "confidence": round(getattr(first_decision, "confidence", 1.0), 3),
                "top_score": getattr(first_decision, "top_score", 100),
                "reason": getattr(first_decision, "reason", ""),
            }
        
        logger.info(f"Researcher: returning {len(docs)} documents (queries={len(expanded_queries)})")
        return {
            "retrieved_docs": docs,
            "retrieval_attempts": attempts + 1,
            "tried_queries": new_tried,
            "retrieval_meta": ret_meta,
        }

researcher_node = ResearcherAgentNode()

