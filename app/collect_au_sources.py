from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlparse

from collections import deque
import time

import fitz
import requests
from bs4 import BeautifulSoup

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - optional runtime dependency
    curl_requests = None

DEFAULT_TARGETS = [
    "https://validex.com.au/faqs.html",
    "https://validex.com.au/how-it-works.html",
    "https://validex.com.au/webapp/#/core/blogs",
    "https://www.acic.gov.au/our-services/national-police-checking-service",
    "https://www.afp.gov.au/",
    "https://www.oaic.gov.au/privacy",
]

ALLOWED_DOMAINS = {
    "validex.com.au",
    "www.validex.com.au",
    "acic.gov.au",
    "www.acic.gov.au",
    "afp.gov.au",
    "www.afp.gov.au",
    "oaic.gov.au",
    "www.oaic.gov.au",
    "nsw.gov.au",
    "www.nsw.gov.au",
    "vic.gov.au",
    "www.vic.gov.au",
    "qld.gov.au",
    "www.qld.gov.au",
}

LEGAL_CORE_KEYWORDS = [
    "police check",
    "background",
    "identity",
    "conviction",
    "legislation",
    "australia",
    "application",
]

AU_POLICE_CHECK_KEYWORDS = [
    "afp",
    "acic",
    "check",
    "applicant",
    "identity",
    "result",
    "conviction",
]

MIN_CHUNK_WORDS = int(os.getenv("COLLECT_MIN_CHUNK_WORDS", "28"))
MIN_KEYWORD_MATCHES = int(os.getenv("COLLECT_MIN_KEYWORD_MATCHES", "2"))
COLLECT_REQUIRE_STEALTH = os.getenv("COLLECT_REQUIRE_STEALTH", "0") == "1"

NOISE_PHRASES = [
    "cookie policy",
    "subscribe now",
    "all rights reserved",
    "follow us on",
    "privacy statement",
    "last updated by admin",
]


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _keyword_match_count(text: str, keywords: list[str] | None = None) -> int:
    lowered = text.lower()
    source = LEGAL_CORE_KEYWORDS if keywords is None else keywords

    seen: set[str] = set()
    hits = 0
    for raw_keyword in source:
        keyword = raw_keyword.strip().lower()
        if not keyword or keyword in seen:
            continue
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        if re.search(pattern, lowered):
            seen.add(keyword)
            hits += 1
    return hits


def _is_quality_chunk(
    chunk: str,
    min_words: int = MIN_CHUNK_WORDS,
    min_keyword_matches: int = MIN_KEYWORD_MATCHES,
) -> bool:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", chunk)
    if len(words) < max(1, min_words):
        return False

    required_hits = max(2, max(1, min_keyword_matches))
    keyword_hits = _keyword_match_count(chunk, keywords=AU_POLICE_CHECK_KEYWORDS)
    return keyword_hits >= required_hits


def _topic_from_url(url: str) -> str:
    lower = url.lower()
    if "privacy" in lower or "oaic" in lower:
        return "privacy"
    if "police" in lower:
        return "police_check"
    if "background" in lower:
        return "background_check"
    if "how-it-works" in lower:
        return "process"
    if "faq" in lower:
        return "faq"
    return "compliance"


def _source_type(url: str) -> str:
    lower = url.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if "faq" in lower:
        return "faq"
    if "blog" in lower:
        return "blog"
    if "how-it-works" in lower:
        return "guide"
    return "webpage"


def _is_allowed(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ALLOWED_DOMAINS


def _has_noise_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in NOISE_PHRASES)


def _is_semantic_paragraph(text: str) -> bool:
    # Require sentence-level substance.
    sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])
    if sentence_count < 2:
        return False

    # Reject ID-like or code-like noise with long repeated symbols.
    if re.search(r"[-*_=#]{4,}", text):
        return False

    letters = re.findall(r"[A-Za-z]", text)
    if not letters:
        return False

    upper = sum(1 for ch in letters if ch.isupper())
    lower = sum(1 for ch in letters if ch.islower())
    if lower == 0:
        return False

    upper_ratio = upper / max(1, upper + lower)
    return upper_ratio <= 0.45


def _select_content_root(soup: BeautifulSoup):
    selectors = [
        "main",
        "article",
        "div[class*='content' i]",
        "div[class*='body' i]",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup.body


def _normalize_discovered_url(url: str, base_url: str) -> str:
    resolved = urljoin(base_url, url)
    resolved = resolved.split("#", 1)[0].strip()
    return resolved.rstrip("/")


def _discover_sub_links(html: str, source_url: str, limit: int = 10) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    root = _select_content_root(soup)
    if root is None:
        return []

    discovered: list[str] = []
    seen: set[str] = set()
    for anchor in root.select("a[href]"):
        href = str(anchor.get("href", "")).strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue

        resolved = _normalize_discovered_url(href, source_url)
        if not resolved or not _is_allowed(resolved):
            continue

        if not re.search(r"/(guidance|advice|decision|report)/", resolved.lower()):
            continue

        if resolved in seen:
            continue

        seen.add(resolved)
        discovered.append(resolved)
        if len(discovered) >= max(1, limit):
            break

    return discovered


def _discover_via_sitemap(sitemap_url: str, limit: int = 10) -> list[str]:
    if COLLECT_REQUIRE_STEALTH and curl_requests is None:
        raise RuntimeError(
            "COLLECT_REQUIRE_STEALTH=1 but curl_cffi is unavailable. Install curl_cffi before crawling production sources."
        )

    request_headers = {
        "Accept": "application/xml,text/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
    }

    if curl_requests is not None:
        response = curl_requests.get(
            sitemap_url,
            impersonate="chrome120",
            timeout=20,
            headers=request_headers,
        )
    else:
        response = requests.get(
            sitemap_url,
            timeout=(10, 30),
            headers={
                **request_headers,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ChatBot-Validex-RAG/1.0",
            },
        )

    if response.status_code == 403:
        raise RuntimeError(
            "Sitemap also returned 403. Vui lòng tải thủ công trang này dưới dạng PDF vào thư mục data/raw/pdfs"
        )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "xml")
    discovered: list[str] = []
    seen: set[str] = set()
    for loc in soup.find_all("loc"):
        url = _normalize_discovered_url(loc.get_text(strip=True), sitemap_url)
        if not url or not _is_allowed(url):
            continue
        lower = url.lower()
        if "/guidance-and-advice/" not in lower and not any(pattern in lower for pattern in ["/guidance/", "/advice/", "/decision/", "/report/"]):
            continue
        if url in seen:
            continue
        seen.add(url)
        discovered.append(url)
        if len(discovered) >= max(1, limit):
            break

    return discovered


def _extract_text_from_html(html: str, source_url: str = "") -> tuple[str, list[dict[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    main = _select_content_root(soup)
    if main is None:
        return "", []

    rejected: list[dict[str, str]] = []

    # Prefer meaningful content blocks over raw page text for cleaner chunks.
    content_nodes = main.select("h1, h2, h3, h4, p, li, blockquote")
    if content_nodes:
        lines = [node.get_text(" ", strip=True) for node in content_nodes]
    else:
        lines = [line.strip() for line in main.get_text("\n", strip=True).splitlines()]

    cleaned_lines: list[str] = []
    for raw_line in lines:
        line = _clean_text(str(raw_line or "").replace("\xa0", " "))
        if not line:
            continue
        if _has_noise_phrase(line):
            rejected.append(
                {
                    "stage": "noise_phrase",
                    "source_url": source_url,
                    "reason": "matched_noise_phrase",
                    "text": line[:320],
                }
            )
            continue
        if not _is_semantic_paragraph(line):
            rejected.append(
                {
                    "stage": "semantic_paragraph",
                    "source_url": source_url,
                    "reason": "paragraph_heuristic_failed",
                    "text": line[:320],
                }
            )
            continue
        cleaned_lines.append(line)

    # Fallback for pages where semantic blocks are sparse or hidden.
    if len(cleaned_lines) < 3:
        body_text = soup.get_text("\n", strip=True)
        fallback_lines = [line.strip() for line in body_text.splitlines() if line.strip()]
        for raw_line in fallback_lines:
            line = _clean_text(str(raw_line or "").replace("\xa0", " "))
            if not line or _has_noise_phrase(line) or not _is_semantic_paragraph(line):
                continue
            cleaned_lines.append(line)

    if not cleaned_lines:
        return "", rejected

    merged = "\n".join(cleaned_lines)
    return _clean_text(merged), rejected


def _extract_text_from_pdf_bytes(payload: bytes) -> str:
    pdf = fitz.open(stream=payload, filetype="pdf")
    pages: list[str] = []
    for page in pdf:
        text = page.get_text() or ""
        text = _clean_text(text)
        if text:
            pages.append(text)
    pdf.close()
    return "\n".join(pages)


def _extract_text_from_pdf_file(pdf_path: Path) -> str:
    pdf = fitz.open(pdf_path)
    pages: list[str] = []
    for page in pdf:
        text = page.get_text() or ""
        text = _clean_text(text)
        if text:
            pages.append(text)
    pdf.close()
    return "\n".join(pages)


def _topic_from_filename(stem: str) -> str:
    lowered = stem.lower()
    if "privacy" in lowered:
        return "privacy"
    if "processing" in lowered or "time" in lowered:
        return "processing_time"
    if "document" in lowered or "requirement" in lowered:
        return "requirements"
    if "police" in lowered:
        return "police_check"
    if "background" in lowered:
        return "background_check"
    return "compliance"


def _source_key_from_pdf_path(pdf_path: Path) -> str:
    return f"file://{pdf_path.resolve().as_posix()}"


def _chunk_text(text: str, chunk_size: int = 1800, overlap: int = 220) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _fetch_url(url: str, timeout: int = 20) -> tuple[str, str, list[dict[str, str]], str]:
    if COLLECT_REQUIRE_STEALTH and curl_requests is None:
        raise RuntimeError(
            "COLLECT_REQUIRE_STEALTH=1 but curl_cffi is unavailable. Install curl_cffi before crawling production sources."
        )

    request_headers = {
        "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
    }

    if curl_requests is not None:
        response = curl_requests.get(
            url,
            impersonate="chrome120",
            timeout=max(10, int(timeout)),
            headers=request_headers,
        )
    else:
        response = requests.get(
            url,
            timeout=(10, timeout + 15),
            headers={
                **request_headers,
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ChatBot-Validex-RAG/1.0",
            },
        )

    if response.status_code == 403:
        raise RuntimeError(
            "403 forbidden. Source blocks bot traffic. Download PDF manually into data/raw/pdfs and ingest locally."
        )

    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if url.lower().endswith(".pdf") or "application/pdf" in content_type:
        text = _extract_text_from_pdf_bytes(response.content)
        return text, "pdf", [], ""

    text, rejected = _extract_text_from_html(response.text, source_url=url)
    return text, "html", rejected, response.text


def _load_state(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    states = payload.get("states", {}) if isinstance(payload, dict) else {}
    return states if isinstance(states, dict) else {}


def _load_existing_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        source_url = str(record.get("source_url", "")).strip()
        if not source_url:
            continue
        grouped.setdefault(source_url, []).append(record)
    return grouped


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def collect_sources(
    target_urls: list[str] | None = None,
    output_jsonl: str = "data/canonical/au_blog_chunks.jsonl",
    output_summary: str = "data/canonical/au_blog_summary.json",
    state_path: str = "data/canonical/source_state.json",
    local_pdf_dir: str = "data/raw/pdfs",
    include_local_pdfs: bool = True,
    incremental: bool = True,
    rejected_output_path: str = "data/samples/rejected_chunks.jsonl",
) -> dict[str, Any]:
    targets = DEFAULT_TARGETS if target_urls is None else target_urls
    valid_targets = [url for url in targets if _is_allowed(url)]

    output_path = Path(output_jsonl)
    summary_path = Path(output_summary)
    state_file = Path(state_path)
    rejected_path = Path(rejected_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    previous_state = _load_state(state_file) if incremental else {}
    existing_by_url = _load_existing_records(output_path) if incremental else {}

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    unchanged_urls = 0
    changed_urls = 0
    local_pdf_total = 0
    local_pdf_processed = 0
    local_pdf_unchanged = 0
    filtered_chunks_total = 0
    rejected_chunks: list[dict[str, str]] = []
    next_state: dict[str, str] = {}
    discovery_exit_reason = "targets_exhausted"
    discovered_sub_links_tried: list[str] = []
    discovered_sub_links_seen: set[str] = set()
    sitemap_seeded = False
    pending_urls = deque((url, 0) for url in valid_targets)
    discovered_chunks_total = 0

    while pending_urls:
        url, depth = pending_urls.popleft()
        if url in next_state:
            continue

        is_discovered_sub_link = depth > 0
        if is_discovered_sub_link:
            if url in discovered_sub_links_seen:
                continue
            if len(discovered_sub_links_tried) >= 10:
                discovery_exit_reason = "sub_links_exhausted"
                break
            discovered_sub_links_seen.add(url)
            discovered_sub_links_tried.append(url)
            time.sleep(random.uniform(2, 5))

        try:
            text, extracted_type, rejected_from_html, raw_html = _fetch_url(url)
            rejected_chunks.extend(rejected_from_html)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            if not sitemap_seeded and "403" in str(exc) and "oaic.gov.au" in url:
                sitemap_seeded = True
                sitemap_url = urljoin(url, "/sitemap.xml")
                try:
                    sitemap_urls = _discover_via_sitemap(sitemap_url, limit=20)
                except Exception as sitemap_exc:
                    errors.append({"url": sitemap_url, "error": str(sitemap_exc)})
                    errors.append(
                        {
                            "url": sitemap_url,
                            "error": "Vui lòng tải thủ công trang này dưới dạng PDF vào thư mục data/raw/pdfs",
                        }
                    )
                else:
                    for sitemap_url_item in sitemap_urls:
                        if sitemap_url_item in next_state:
                            continue
                        if sitemap_url_item not in discovered_sub_links_seen:
                            pending_urls.append((sitemap_url_item, depth + 1))
            continue

        if not text:
            errors.append({"url": url, "error": "empty content"})
            continue

        current_hash = _content_hash(text)
        next_state[url] = current_hash

        if incremental and previous_state.get(url) == current_hash and url in existing_by_url:
            records.extend(existing_by_url[url])
            if is_discovered_sub_link:
                discovered_chunks_total += len(existing_by_url[url])
                if discovered_chunks_total >= 20:
                    discovery_exit_reason = "target_chunk_threshold_reached"
                    break
            else:
                unchanged_urls += 1
            continue

        if not is_discovered_sub_link:
            changed_urls += 1

        title = urlparse(url).path.strip("/") or urlparse(url).netloc
        source_type = _source_type(url)
        if extracted_type == "pdf":
            source_type = "pdf"

        chunks = _chunk_text(text)
        for idx, chunk in enumerate(chunks, start=1):
            if not _is_quality_chunk(chunk):
                filtered_chunks_total += 1
                rejected_chunks.append(
                    {
                        "stage": "quality_chunk",
                        "source_url": url,
                        "reason": "au_police_keyword_threshold",
                        "text": chunk[:320],
                    }
                )
                continue
            hash_key = hashlib.sha1(f"{url}:{idx}:{chunk[:120]}".encode("utf-8")).hexdigest()[:16]
            records.append(
                {
                    "doc_id": f"doc_{hash_key}",
                    "chunk_id": f"chunk_{hash_key}_{idx}",
                    "source_url": url,
                    "source_domain": urlparse(url).netloc.lower(),
                    "source_type": source_type,
                    "topic": _topic_from_url(url),
                    "region": "AU",
                    "title": title,
                    "authority_score": 0.95 if "gov.au" in url else 0.8,
                    "approved": True,
                    "text": chunk,
                }
            )
            if is_discovered_sub_link:
                discovered_chunks_total += 1
                if discovered_chunks_total >= 20:
                    discovery_exit_reason = "target_chunk_threshold_reached"
                    break

        if discovery_exit_reason == "target_chunk_threshold_reached":
            break

        if extracted_type == "html" and raw_html:
            sub_links = _discover_sub_links(raw_html, url, limit=10)
            for sub_link in sub_links:
                if sub_link in discovered_sub_links_seen or sub_link in next_state:
                    continue
                if len(discovered_sub_links_seen) >= 10:
                    discovery_exit_reason = "sub_links_exhausted"
                    break
                pending_urls.append((sub_link, depth + 1))

            if not sitemap_seeded and "oaic.gov.au" in url.lower():
                sitemap_seeded = True
                sitemap_url = urljoin(url, "/sitemap.xml")
                try:
                    sitemap_urls = _discover_via_sitemap(sitemap_url, limit=20)
                except Exception as sitemap_exc:
                    errors.append({"url": sitemap_url, "error": str(sitemap_exc)})
                    errors.append(
                        {
                            "url": sitemap_url,
                            "error": "Vui lòng tải thủ công trang này dưới dạng PDF vào thư mục data/raw/pdfs",
                        }
                    )
                else:
                    for sitemap_url_item in sitemap_urls:
                        if sitemap_url_item in next_state:
                            continue
                        pending_urls.append((sitemap_url_item, depth + 1))

        if discovery_exit_reason == "sub_links_exhausted":
            break

    if include_local_pdfs and discovery_exit_reason != "target_chunk_threshold_reached":
        pdf_root = Path(local_pdf_dir)
        if pdf_root.exists():
            for pdf_path in sorted(pdf_root.glob("*.pdf")):
                local_pdf_total += 1
                source_key = _source_key_from_pdf_path(pdf_path)
                try:
                    text = _extract_text_from_pdf_file(pdf_path)
                except Exception as exc:
                    errors.append({"url": source_key, "error": str(exc)})
                    continue

                if not text:
                    errors.append({"url": source_key, "error": "empty content"})
                    continue

                current_hash = _content_hash(text)
                next_state[source_key] = current_hash

                if incremental and previous_state.get(source_key) == current_hash and source_key in existing_by_url:
                    records.extend(existing_by_url[source_key])
                    local_pdf_unchanged += 1
                    continue

                local_pdf_processed += 1

                chunks = _chunk_text(text)
                source_domain = pdf_path.parent.name or "local"
                title = pdf_path.stem.replace("_", " ").strip()
                topic = _topic_from_filename(pdf_path.stem)

                for idx, chunk in enumerate(chunks, start=1):
                    if not _is_quality_chunk(chunk):
                        filtered_chunks_total += 1
                        rejected_chunks.append(
                            {
                                "stage": "quality_chunk",
                                "source_url": source_key,
                                "reason": "au_police_keyword_threshold",
                                "text": chunk[:320],
                            }
                        )
                        continue
                    hash_key = hashlib.sha1(
                        f"{source_key}:{idx}:{chunk[:120]}".encode("utf-8")
                    ).hexdigest()[:16]
                    records.append(
                        {
                            "doc_id": f"doc_{hash_key}",
                            "chunk_id": f"chunk_{hash_key}_{idx}",
                            "source_url": source_key,
                            "source_domain": source_domain,
                            "source_type": "pdf",
                            "topic": topic,
                            "region": "AU",
                            "title": title,
                            "authority_score": 0.85,
                            "approved": True,
                            "text": chunk,
                        }
                    )

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with rejected_path.open("w", encoding="utf-8") as f:
        for record in rejected_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    state_file.write_text(json.dumps({"states": next_state}, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "targets_total": len(targets),
        "targets_allowed": len(valid_targets),
        "chunks_total": len(records),
        "changed_urls": changed_urls,
        "unchanged_urls": unchanged_urls,
        "local_pdf_total": local_pdf_total,
        "local_pdf_processed": local_pdf_processed,
        "local_pdf_unchanged": local_pdf_unchanged,
        "filtered_chunks_total": filtered_chunks_total,
        "rejected_chunks_total": len(rejected_chunks),
        "rejected_output_path": str(rejected_path),
        "discovery_exit_reason": discovery_exit_reason,
        "discovered_sub_links_tried": discovered_sub_links_tried,
        "discovered_sub_links_total": len(discovered_sub_links_tried),
        "discovered_chunks_total": discovered_chunks_total,
        "cleaning_rules": {
            "min_chunk_words": MIN_CHUNK_WORDS,
            "min_keyword_matches": max(2, MIN_KEYWORD_MATCHES),
            "core_keywords": LEGAL_CORE_KEYWORDS,
            "au_police_check_keywords": AU_POLICE_CHECK_KEYWORDS,
            "noise_phrases": NOISE_PHRASES,
            "require_stealth": COLLECT_REQUIRE_STEALTH,
        },
        "errors_total": len(errors),
        "errors": errors,
        "output_jsonl": str(output_path),
        "state_path": str(state_file),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = collect_sources()
    print(json.dumps(result, indent=2))
