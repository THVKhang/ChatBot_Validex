"""Web scraper utility — extracts clean text from URLs for the researcher node."""
from __future__ import annotations

import logging
import re
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Domains to never scrape
BLOCKED_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "linkedin.com", "reddit.com", "youtube.com", "tiktok.com",
    "pinterest.com", "quora.com",
}


def _extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML without BeautifulSoup dependency."""
    # Remove script and style blocks
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    # Remove nav, header, footer, aside
    for tag in ["nav", "header", "footer", "aside", "iframe"]:
        text = re.sub(rf"<{tag}[^>]*>[\s\S]*?</{tag}>", "", text, flags=re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_url(url: str, timeout: int = 8, max_chars: int = 5000) -> str | None:
    """Scrape a URL and return clean text content.
    
    Returns None if scraping fails or URL is blocked.
    """
    from urllib.parse import urlparse
    
    domain = urlparse(url).netloc.lower().lstrip("www.")
    if domain in BLOCKED_DOMAINS:
        logger.debug(f"Scraper: blocked domain {domain}")
        return None

    try:
        req = Request(url, headers=_HEADERS)
        with urlopen(req, timeout=timeout) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None
            raw = resp.read(200_000)  # Max 200KB
            charset = "utf-8"
            ct_match = re.search(r"charset=([^\s;]+)", content_type)
            if ct_match:
                charset = ct_match.group(1)
            html = raw.decode(charset, errors="ignore")
    except (URLError, TimeoutError, OSError, Exception) as exc:
        logger.debug(f"Scraper: failed to fetch {url}: {exc}")
        return None

    text = _extract_text_from_html(html)
    if len(text) < 100:
        return None
    
    return text[:max_chars]


def scrape_multiple(urls: list[str], max_urls: int = 3, timeout: int = 8) -> list[dict]:
    """Scrape multiple URLs, return list of {url, text} dicts.
    
    Stops after max_urls successful scrapes.
    """
    results = []
    for url in urls:
        if len(results) >= max_urls:
            break
        text = scrape_url(url, timeout=timeout)
        if text:
            results.append({"url": url, "text": text})
            logger.info(f"Scraper: scraped {url} ({len(text)} chars)")
    return results
