"""Smart Researcher Node — multi-query retrieval, deep scraping, LLM summarization."""
import json
import logging
import re
from langchain_core.documents import Document
from app.graph_state import GraphState, RetrievedDoc
from app.langchain_pipeline import pipeline
from app.config import settings

logger = logging.getLogger(__name__)


def _expand_query_with_llm(topic: str) -> list[str]:
    """Use LLM to generate 3 diverse sub-queries for better retrieval coverage."""
    if pipeline._llm is None:
        return [topic]

    prompt = (
        "You are a search query optimizer. Given a topic, generate 3 diverse search queries "
        "that would help find comprehensive information about it.\n\n"
        "Rules:\n"
        "- Each query should approach the topic from a different angle\n"
        "- Keep queries specific and search-engine-friendly\n"
        "- Include relevant keywords and synonyms\n"
        "- Focus on Australian context if the topic relates to policy/compliance\n\n"
        f"Topic: {topic}\n\n"
        "Return ONLY a JSON array of 3 strings, e.g.:\n"
        '["query 1", "query 2", "query 3"]'
    )

    try:
        response = pipeline._llm.invoke(prompt)
        raw = getattr(response, "content", str(response))
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            queries = json.loads(match.group(0))
            if isinstance(queries, list) and len(queries) >= 2:
                logger.info(f"Query expansion: {topic} → {queries}")
                return [topic] + [str(q) for q in queries[:3]]
    except Exception as exc:
        logger.warning(f"Query expansion failed: {exc}")

    return [topic]


def _llm_summarize_content(text: str, topic: str, max_output: int = 500) -> str:
    """Use LLM to create a focused summary of scraped content."""
    if pipeline._llm is None or not text:
        return text[:max_output]

    prompt = (
        f"Summarize the following text focusing on information relevant to: {topic}\n"
        "Keep the summary factual, concise, and preserve key data points, statistics, and citations.\n"
        f"Maximum {max_output} characters.\n\n"
        f"Text:\n{text[:3000]}\n\n"
        "Summary:"
    )

    try:
        response = pipeline._llm.invoke(prompt)
        summary = getattr(response, "content", str(response)).strip()
        if len(summary) > 50:
            return summary[:max_output]
    except Exception as exc:
        logger.warning(f"Content summarization failed: {exc}")

    return text[:max_output]


def _score_doc_relevance(doc_text: str, topic: str) -> float:
    """Quick heuristic relevance score (0-1) without calling LLM."""
    topic_words = set(re.findall(r"\w+", topic.lower()))
    doc_words = set(re.findall(r"\w+", doc_text[:500].lower()))
    if not topic_words:
        return 0.5
    overlap = len(topic_words & doc_words)
    return min(1.0, overlap / max(1, len(topic_words)))


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
            summary = _llm_summarize_content(item["text"], topic)
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


def researcher_node(state: GraphState) -> GraphState:
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
        return {"retrieved_docs": docs}
        
    # --- Multi-query retrieval ---
    expanded_queries = _expand_query_with_llm(topic)
    all_documents: list[Document] = []
    
    for query in expanded_queries:
        payload = {"effective_topic": query, "retrieval_top_k": settings.top_k}
        try:
            bundle = pipeline._retrieve(payload)
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
    
    if len(good_docs) < 2:
        logger.info(f"Researcher: only {len(good_docs)} good docs, triggering web search + scraping")
        web_docs = _web_search_with_scraping(topic, max_results=3)
        all_documents = good_docs + web_docs
    else:
        all_documents = good_docs

    # Limit to top 5
    all_documents = all_documents[:5]
    
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

    logger.info(f"Researcher: returning {len(docs)} documents (queries={len(expanded_queries)})")
    return {"retrieved_docs": docs}
