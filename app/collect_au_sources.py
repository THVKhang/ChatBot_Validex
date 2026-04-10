from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

import fitz
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def _extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.select_one("div[role='main']")
        or soup.select_one("#content")
        or soup.select_one(".content")
        or soup.select_one(".main-content")
        or soup.body
    )
    if main is None:
        return ""

    # Prefer meaningful content blocks over raw page text for cleaner chunks.
    content_nodes = main.select("h1, h2, h3, h4, p, li, blockquote")
    if content_nodes:
        lines = [node.get_text(" ", strip=True) for node in content_nodes]
    else:
        lines = [line.strip() for line in main.get_text("\n", strip=True).splitlines()]

    lines = [line for line in lines if line]

    # Fallback for pages where semantic blocks are sparse or hidden.
    if len(lines) < 5:
        body_text = soup.get_text("\n", strip=True)
        fallback_lines = [line.strip() for line in body_text.splitlines() if line.strip()]
        lines = fallback_lines if len(fallback_lines) > len(lines) else lines

    merged = "\n".join(lines)
    return _clean_text(merged)


def _build_http_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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


def _fetch_url(url: str, timeout: int = 20) -> tuple[str, str]:
    session = _build_http_session()
    response = session.get(
        url,
        timeout=(10, timeout + 15),
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 ChatBot-Validex-RAG/1.0",
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-AU,en;q=0.9",
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
        return text, "pdf"

    text = _extract_text_from_html(response.text)
    return text, "html"


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
) -> dict[str, Any]:
    targets = DEFAULT_TARGETS if target_urls is None else target_urls
    valid_targets = [url for url in targets if _is_allowed(url)]

    output_path = Path(output_jsonl)
    summary_path = Path(output_summary)
    state_file = Path(state_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    previous_state = _load_state(state_file) if incremental else {}
    existing_by_url = _load_existing_records(output_path) if incremental else {}

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    unchanged_urls = 0
    changed_urls = 0
    local_pdf_total = 0
    local_pdf_processed = 0
    local_pdf_unchanged = 0
    next_state: dict[str, str] = {}

    for url in valid_targets:
        try:
            text, extracted_type = _fetch_url(url)
        except Exception as exc:
            errors.append({"url": url, "error": str(exc)})
            continue

        if not text:
            errors.append({"url": url, "error": "empty content"})
            continue

        current_hash = _content_hash(text)
        next_state[url] = current_hash

        if incremental and previous_state.get(url) == current_hash and url in existing_by_url:
            records.extend(existing_by_url[url])
            unchanged_urls += 1
            continue

        changed_urls += 1

        title = urlparse(url).path.strip("/") or urlparse(url).netloc
        source_type = _source_type(url)
        if extracted_type == "pdf":
            source_type = "pdf"

        chunks = _chunk_text(text)
        for idx, chunk in enumerate(chunks, start=1):
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

    if include_local_pdfs:
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
