"""AI Source Discovery Agent — Uses LLM + DuckDuckGo Search to find new data sources."""
from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

from app.config import settings

logger = logging.getLogger(__name__)

# Topics the system covers — used to generate search queries
DISCOVERY_TOPICS = [
    "Australian National Police Check process",
    "Working With Children Check WWCC Australia",
    "NDIS worker screening check Australia",
    "Aged care worker screening Australia",
    "Australian background check for employment",
    "Australian privacy law employee screening",
    "Fair Work Act background checks",
    "Australian immigration work visa checks",
    "State police check NSW VIC QLD SA WA",
    "Australian criminal history check legislation",
    "Volunteer screening requirements Australia",
    "Right to work verification Australia",
    "Australian PEP check politically exposed persons requirements",
    "VEVO work rights check Australia immigration",
    "Digital identity verification framework TDIF Australia",
    "Reference checks compliance Australia Fair Work",
    "KYC AML background check requirements Australia",
    "Fit and Proper person test Australian guidelines",
]

# Domains already known — we skip these to find NEW sources
KNOWN_DOMAINS = {
    "validex.com.au",
    "www.validex.com.au",
    "acic.gov.au",
    "www.acic.gov.au",
    "afp.gov.au",
    "www.afp.gov.au",
    "oaic.gov.au",
    "www.oaic.gov.au",
}

# Domains to always reject (social media, forums, etc.)
REJECTED_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "linkedin.com", "reddit.com", "youtube.com", "tiktok.com",
    "pinterest.com", "quora.com", "wikipedia.org",
}


def _duckduckgo_search(query: str, num: int = 5) -> list[dict[str, str]]:
    """Search using DuckDuckGo (free, no API key needed)."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS  # type: ignore[import-not-found]
        except ImportError:
            logger.error("ddgs not installed. Run: pip install ddgs")
            return []

    try:
        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, region="au-en", max_results=num))
    except Exception as exc:
        logger.error("DuckDuckGo search failed for query '%s': %s", query, exc)
        return []

    results = []
    for item in raw_results:
        url = str(item.get("href", "")).strip()
        if not url:
            continue
        domain = urlparse(url).netloc.lower().lstrip("www.")
        if domain in REJECTED_DOMAINS:
            continue
        results.append({
            "title": str(item.get("title", "")),
            "url": url,
            "snippet": str(item.get("body", "")),
        })
    return results


def _google_custom_search(query: str, num: int = 5) -> list[dict[str, str]]:
    """Call Google Custom Search JSON API (fallback, requires API key)."""
    from urllib.parse import quote_plus
    from urllib.request import Request, urlopen

    api_key = settings.google_search_api_key.strip()
    cx = settings.google_search_cx.strip()
    if not api_key or not cx:
        return []

    endpoint = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?key={api_key}"
        f"&cx={cx}"
        f"&q={quote_plus(query)}"
        f"&num={min(num, 10)}"
        f"&gl=au"
        f"&lr=lang_en"
    )

    try:
        req = Request(endpoint, headers={"Accept": "application/json"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("Google Custom Search failed for query '%s': %s", query, exc)
        return []

    results = []
    for item in data.get("items", []):
        url = str(item.get("link", "")).strip()
        if not url:
            continue
        domain = urlparse(url).netloc.lower().lstrip("www.")
        if domain in REJECTED_DOMAINS:
            continue
        results.append({
            "title": str(item.get("title", "")),
            "url": url,
            "snippet": str(item.get("snippet", "")),
        })
    return results


def _search(query: str, num: int = 5) -> list[dict[str, str]]:
    """Search using DuckDuckGo (primary) or Google Custom Search (fallback)."""
    results = _duckduckgo_search(query, num)
    if results:
        return results
    # Fallback to Google if DuckDuckGo fails
    return _google_custom_search(query, num)


def _semantic_evaluate_relevance(url: str, title: str, snippet: str) -> dict[str, Any]:
    """Score relevance using Local Semantics (0 LLM tokens)."""
    try:
        from app.local_semantics import get_embedding, get_reference_embedding, cosine_similarity
        
        text = f"{title} {snippet} {url}"
        
        text_emb = get_embedding(text)
        ref_emb = get_reference_embedding()
        
        sim = cosine_similarity(text_emb, ref_emb)
        
        # Map similarity (typically 0.1 to 0.5) to a 1-10 score
        # Using a conservative mapping
        raw_score = (sim * 20)
        
        # .gov.au domains get a bonus
        if ".gov.au" in url.lower():
            raw_score += 2
            
        score = min(10, max(1, int(round(raw_score))))
        
        return {
            "score": score,
            "reason": f"semantic_sim={sim:.3f}",
        }
        
    except Exception as exc:
        logger.error(f"Semantic relevance evaluation failed: {exc}")
        return {"score": 1, "reason": "evaluation_failed"}


def discover_new_sources(
    existing_urls: set[str] | None = None,
    max_new_urls: int | None = None,
    min_relevance: int | None = None,
) -> dict[str, Any]:
    """
    Run the AI-powered source discovery pipeline.

    Returns a summary dict with 'approved_urls', 'rejected_urls', 'errors'.
    """
    limit = max_new_urls or settings.discovery_max_new_urls
    threshold = min_relevance or settings.discovery_min_relevance_score
    known = existing_urls or set()

    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_urls: set[str] = set()

    logger.info("Discovery Agent starting. Topics: %d, limit: %d, threshold: %d",
                len(DISCOVERY_TOPICS), limit, threshold)

    for topic in DISCOVERY_TOPICS:
        if len(approved) >= limit:
            break

        search_results = _search(topic, num=5)
        if not search_results:
            continue

        for result in search_results:
            if len(approved) >= limit:
                break

            url = result["url"]
            domain = urlparse(url).netloc.lower()

            # Skip already-known or already-seen URLs
            if url in known or url in seen_urls:
                continue
            seen_urls.add(url)

            # Skip known domains (we already crawl them)
            clean_domain = domain.lstrip("www.")
            if clean_domain in KNOWN_DOMAINS:
                continue

            # Semantic-based evaluation (0 LLM tokens)
            evaluation = _semantic_evaluate_relevance(url, result["title"], result["snippet"])
            score = evaluation.get("score", 0)

            entry = {
                "url": url,
                "domain": domain,
                "title": result["title"],
                "snippet": result["snippet"],
                "ai_score": score,
                "ai_reason": evaluation.get("reason", ""),
                "topic_query": topic,
            }

            if score >= threshold:
                approved.append(entry)
                logger.info("✅ Approved: %s (score=%d)", url, score)
            else:
                rejected.append(entry)
                logger.debug("❌ Rejected: %s (score=%d)", url, score)

    summary = {
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "approved_urls": approved,
        "rejected_urls": rejected,
        "errors": errors,
    }

    logger.info("Discovery Agent finished. Approved: %d, Rejected: %d",
                len(approved), len(rejected))

    return summary
