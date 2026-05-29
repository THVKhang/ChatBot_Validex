from __future__ import annotations

import hashlib
import json
import logging
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

logger = logging.getLogger(__name__)

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover - optional runtime dependency
    curl_requests = None

DEFAULT_TARGETS = [
    # Validex core
    "https://validex.com.au/faqs.html",
    "https://validex.com.au/how-it-works.html",
    "https://validex.com.au/webapp/#/core/blogs",
    # Federal agencies
    "https://www.acic.gov.au/our-services/national-police-checking-service",
    "https://www.afp.gov.au/",
    "https://www.oaic.gov.au/privacy",
    # State-level police checks
    "https://www.police.nsw.gov.au/online_services/national_police_check",
    "https://www.police.vic.gov.au/national-police-check",
    "https://www.police.qld.gov.au/units/criminal-history-screening",
    # WWCC (Working With Children Check)
    "https://www.kidsguardian.nsw.gov.au/working-with-children",
    "https://www.workingwithchildren.vic.gov.au/",
    # Aged Care & NDIS
    "https://www.ndiscommission.gov.au/workers/worker-screening",
    "https://www.agedcarequality.gov.au/",
    # Fair Work & Immigration
    "https://www.fairwork.gov.au/",
    "https://immi.homeaffairs.gov.au/visas/working-in-australia",
    # ── Primary Legislation Sources (The Golden Sources) ──
    # Federal Register of Legislation
    "https://www.legislation.gov.au/C2004A01364/latest/text",  # Crimes Act 1914 (Cth) — Part VIIC Spent Convictions
    "https://www.legislation.gov.au/C2004A03712/latest/text",  # Privacy Act 1988 (Cth)
    "https://www.legislation.gov.au/C2004A01389/latest/text",  # Australian Federal Police Act 1979 (Cth)
    # NSW Legislation
    "https://legislation.nsw.gov.au/view/html/inforce/current/act-1991-008",  # Criminal Records Act 1991 (NSW)
    "https://legislation.nsw.gov.au/view/html/inforce/current/act-1998-009",  # Child Protection (WWCC) Act 2012 (NSW)
    # VIC Legislation
    "https://www.legislation.vic.gov.au/in-force/acts/spent-convictions-act-2021",  # Spent Convictions Act 2021 (Vic)
    # QLD Legislation
    "https://www.legislation.qld.gov.au/view/html/inforce/current/act-2004-015",  # Criminal Law (Rehabilitation) Act 1986 (QLD)
]

ALLOWED_DOMAINS = {
    # Validex
    "validex.com.au",
    "www.validex.com.au",
    # Federal
    "acic.gov.au", "www.acic.gov.au",
    "afp.gov.au", "www.afp.gov.au",
    "oaic.gov.au", "www.oaic.gov.au",
    # State police
    "police.nsw.gov.au", "www.police.nsw.gov.au",
    "police.vic.gov.au", "www.police.vic.gov.au",
    "police.qld.gov.au", "www.police.qld.gov.au",
    "police.sa.gov.au", "www.police.sa.gov.au",
    "police.wa.gov.au", "www.police.wa.gov.au",
    # State gov portals
    "nsw.gov.au", "www.nsw.gov.au",
    "vic.gov.au", "www.vic.gov.au",
    "qld.gov.au", "www.qld.gov.au",
    "sa.gov.au", "www.sa.gov.au",
    "wa.gov.au", "www.wa.gov.au",
    # WWCC
    "kidsguardian.nsw.gov.au", "www.kidsguardian.nsw.gov.au",
    "workingwithchildren.vic.gov.au", "www.workingwithchildren.vic.gov.au",
    # NDIS & Aged Care
    "ndiscommission.gov.au", "www.ndiscommission.gov.au",
    "agedcarequality.gov.au", "www.agedcarequality.gov.au",
    # Fair Work & Immigration
    "fairwork.gov.au", "www.fairwork.gov.au",
    "homeaffairs.gov.au", "immi.homeaffairs.gov.au",
    # ── Legislation Sources ──
    "legislation.gov.au", "www.legislation.gov.au",
    "legislation.nsw.gov.au", "www.legislation.nsw.gov.au",
    "legislation.vic.gov.au", "www.legislation.vic.gov.au",
    "legislation.qld.gov.au", "www.legislation.qld.gov.au",
    "legislation.sa.gov.au", "www.legislation.sa.gov.au",
    "legislation.wa.gov.au", "www.legislation.wa.gov.au",
    "legislation.tas.gov.au", "www.legislation.tas.gov.au",
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
    # Original
    "afp", "acic", "check", "applicant", "identity", "result", "conviction",
    # Expanded — compliance & screening
    "screening", "compliance", "clearance", "criminal", "history",
    "verification", "employment", "disclosure", "legislation", "regulation",
    # WWCC, NDIS, Aged Care
    "wwcc", "children", "ndis", "aged care", "worker",
    # Immigration & Fair Work
    "visa", "right to work", "fair work", "workplace",
    # Privacy
    "privacy", "data protection", "spent conviction",
    # Legal terms (for legislation pages)
    "section", "subsection", "act", "part", "division", "offence",
    "penalty", "rehabilitation", "spent", "waiting period",
]

MIN_CHUNK_WORDS = int(os.getenv("COLLECT_MIN_CHUNK_WORDS", "28"))
MIN_KEYWORD_MATCHES = int(os.getenv("COLLECT_MIN_KEYWORD_MATCHES", "1"))
COLLECT_REQUIRE_STEALTH = os.getenv("COLLECT_REQUIRE_STEALTH", "0") == "1"

NOISE_PHRASES = [
    "cookie policy",
    "subscribe now",
    "all rights reserved",
    "follow us on",
    "privacy statement",
    "last updated by admin",
]


def _ai_evaluate_chunk(chunk: str) -> dict:
    """Score a chunk's relevance using Local Semantics (0 LLM tokens)."""
    try:
        from app.local_semantics import get_embedding, get_reference_embedding, cosine_similarity
        
        chunk_emb = get_embedding(chunk[:1000])
        ref_emb = get_reference_embedding()
        
        sim = cosine_similarity(chunk_emb, ref_emb)
        
        # Map similarity to 1-10 scale
        # Typically similarity > 0.25 is relevant, > 0.4 is highly relevant
        if sim >= 0.4:
            score = 10
        elif sim >= 0.3:
            score = 8
        elif sim >= 0.2:
            score = 6
        elif sim >= 0.1:
            score = 4
        else:
            score = 1
            
        return {
            "score": score,
            "reason": f"semantic_sim={sim:.3f}"
        }
    except Exception as exc:
        return {"score": 5, "reason": f"semantic_evaluation_failed: {exc}"}


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
    if "spent" in lower or "rehabilitation" in lower:
        return "spent_convictions"
    if "crimes-act" in lower or "C2004A01364" in lower:
        return "spent_convictions"
    if "criminal-records" in lower or "act-1991-008" in lower:
        return "spent_convictions"
    if "police" in lower:
        return "police_check"
    if "background" in lower:
        return "background_check"
    if "how-it-works" in lower:
        return "process"
    if "faq" in lower:
        return "faq"
    if "legislation" in lower:
        return "legislation"
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
    if _is_legislation_url(lower):
        return "legislation"
    return "webpage"


# ── Legislation Detection & Metadata ──────────────────────────
def _is_legislation_url(url: str) -> bool:
    """Check if URL is a primary legislation source."""
    lower = url.lower() if isinstance(url, str) else ""
    legislation_domains = [
        "legislation.gov.au", "legislation.nsw.gov.au",
        "legislation.vic.gov.au", "legislation.qld.gov.au",
        "legislation.sa.gov.au", "legislation.wa.gov.au",
        "legislation.tas.gov.au",
    ]
    return any(domain in lower for domain in legislation_domains)


def _detect_jurisdiction(url: str) -> str:
    """Detect jurisdiction from legislation URL."""
    lower = url.lower()
    if "legislation.nsw.gov.au" in lower:
        return "NSW"
    if "legislation.vic.gov.au" in lower:
        return "VIC"
    if "legislation.qld.gov.au" in lower:
        return "QLD"
    if "legislation.sa.gov.au" in lower:
        return "SA"
    if "legislation.wa.gov.au" in lower:
        return "WA"
    if "legislation.tas.gov.au" in lower:
        return "TAS"
    if "legislation.gov.au" in lower:
        return "Commonwealth"
    # Detect from police/gov URLs
    if "nsw.gov.au" in lower:
        return "NSW"
    if "vic.gov.au" in lower:
        return "VIC"
    if "qld.gov.au" in lower:
        return "QLD"
    return "Commonwealth"


def _detect_act_name(url: str, text: str = "") -> str:
    """Detect Act name from URL path or page content."""
    # Known Act mappings by legislation.gov.au ID
    act_mappings = {
        "C2004A01364": "Crimes Act 1914",
        "C2004A03712": "Privacy Act 1988",
        "C2004A01389": "Australian Federal Police Act 1979",
        "act-1991-008": "Criminal Records Act 1991",
        "act-1998-009": "Child Protection (Working with Children) Act 2012",
        "act-2004-015": "Criminal Law (Rehabilitation of Offenders) Act 1986",
    }
    for key, name in act_mappings.items():
        if key in url:
            return name
    if "spent-convictions" in url.lower():
        return "Spent Convictions Act 2021"
    # Try to extract from page content
    match = re.search(r"(?:^|\n)\s*(.+?Act\s+\d{4})", text[:500])
    if match:
        return match.group(1).strip()
    return ""


def _chunk_legal_text(text: str, url: str) -> list[dict]:
    """Split legal text using hierarchical LegalChunker (AST-based).
    
    Delegates to app.legal_chunker.LegalChunker which preserves:
    - Act → Part → Division → Section hierarchy
    - Context enrichment (breadcrumb prefix on every chunk)
    - Section boundary integrity (never cuts mid-section)
    
    Returns list of dicts compatible with the existing collector pipeline.
    """
    jurisdiction = _detect_jurisdiction(url)
    act_name = _detect_act_name(url, text)
    
    try:
        from app.legal_chunker import chunk_legal_text as _legal_chunk
        
        legal_chunks = _legal_chunk(
            text=text,
            act_name=act_name,
            jurisdiction=jurisdiction,
            source_url=url,
        )
        
        if not legal_chunks:
            return []
        
        # Convert LegalChunk objects to dicts for the collector pipeline
        result = []
        for lc in legal_chunks:
            result.append({
                "text": lc.text,            # Includes breadcrumb prefix
                "section": lc.section_number,
                "section_title": lc.section_title,
                "section_ref": lc.section_ref,
                "parent_context": lc.breadcrumb,
                "sub_chunk": lc.chunk_index,
            })
        return result
        
    except ImportError:
        logger.warning("LegalChunker not available — falling back to regex chunking")
        # Fallback: basic regex chunking (original behavior)
        section_pattern = re.compile(
            r'^\s*(?:'
            r'(?:Section\s+)?(\d+[A-Z]*)\s{2,}(.+)'
            r'|(?:Part\s+[IVXLC]+[A-Z]*)\s*[-—]?\s*(.+)'
            r'|(?:Division\s+\d+)\s*[-—]?\s*(.+)'
            r')\s*$',
            re.MULTILINE
        )
        splits = list(section_pattern.finditer(text))
        if not splits:
            return []
        
        chunks = []
        for i, match in enumerate(splits):
            start = match.start()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            chunk_text = text[start:end].strip()
            if len(chunk_text.split()) < 10:
                continue
            section_num = match.group(1) or ""
            section_title = (match.group(2) or match.group(3) or match.group(4) or "").strip()
            if len(chunk_text) > 3000:
                sub_chunks = _chunk_text(chunk_text, chunk_size=2500, overlap=200)
                for j, sub in enumerate(sub_chunks):
                    chunks.append({"text": sub, "section": section_num, "section_title": section_title, "sub_chunk": j + 1})
            else:
                chunks.append({"text": chunk_text, "section": section_num, "section_title": section_title, "sub_chunk": 0})
        return chunks


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


from abc import ABC, abstractmethod

class ContentParserStrategy(ABC):
    """Abstract base class for content parsing strategies."""
    @abstractmethod
    def parse(self, content: Any, source_url: str = "") -> Any:
        pass


class TableParserStrategy(ContentParserStrategy):
    """Strategy for parsing HTML tables and converting them to Markdown."""
    def parse(self, content: Any, source_url: str = "") -> str:
        rows = content.find_all("tr")
        if not rows:
            return ""
        
        md_rows = []
        
        # 1. Process header (or first row)
        headers = []
        first_row = rows[0]
        cols = first_row.find_all(["th", "td"])
        for col in cols:
            headers.append(col.get_text(" ", strip=True))
        
        if not headers:
            return ""
            
        md_rows.append("| " + " | ".join(headers) + " |")
        # Add separator row
        md_rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # 2. Process body rows
        for row in rows[1:]:
            cols = row.find_all(["th", "td"])
            row_cells = []
            for col in cols:
                row_cells.append(col.get_text(" ", strip=True))
            # Match column count
            if len(row_cells) < len(headers):
                row_cells.extend([""] * (len(headers) - len(row_cells)))
            elif len(row_cells) > len(headers):
                row_cells = row_cells[:len(headers)]
            md_rows.append("| " + " | ".join(row_cells) + " |")
            
        return "\n" + "\n".join(md_rows) + "\n"


class HtmlParserStrategy(ContentParserStrategy):
    """Strategy for extracting clean text and tables from HTML content."""
    def __init__(self, table_parser: ContentParserStrategy | None = None):
        self.table_parser = table_parser or TableParserStrategy()

    def parse(self, content: str, source_url: str = "") -> tuple[str, list[dict[str, str]]]:
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
            tag.decompose()

        main = _select_content_root(soup)
        if main is None:
            return "", []

        rejected: list[dict[str, str]] = []

        # Prefer meaningful content blocks over raw page text for cleaner chunks.
        # We now explicitly include 'table' elements to extract structured data tables.
        content_nodes = main.select("h1, h2, h3, h4, p, li, blockquote, table")
        lines: list[str] = []
        if content_nodes:
            for node in content_nodes:
                if node.name == "table":
                    table_md = self.table_parser.parse(node, source_url)
                    if table_md:
                        lines.append(table_md)
                else:
                    lines.append(node.get_text(" ", strip=True))
        else:
            lines = [line.strip() for line in main.get_text("\n", strip=True).splitlines()]

        cleaned_lines: list[str] = []
        for raw_line in lines:
            if raw_line.strip().startswith("|"):
                cleaned_lines.append(raw_line)
                continue
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
                if raw_line.strip().startswith("|"):
                    cleaned_lines.append(raw_line)
                    continue
                line = _clean_text(str(raw_line or "").replace("\xa0", " "))
                if not line or _has_noise_phrase(line) or not _is_semantic_paragraph(line):
                    continue
                cleaned_lines.append(line)

        if not cleaned_lines:
            return "", rejected

        merged = "\n".join(cleaned_lines)
        return _clean_text(merged), rejected


class PdfParserStrategy(ContentParserStrategy):
    """Strategy for extracting text from PDF (supports bytes, path string, or Path)."""
    def parse(self, content: bytes | Path | str, source_url: str = "") -> str:
        if isinstance(content, bytes):
            pdf = fitz.open(stream=content, filetype="pdf")
        else:
            pdf = fitz.open(str(content))
        pages: list[str] = []
        for page in pdf:
            text = page.get_text() or ""
            text = _clean_text(text)
            if text:
                pages.append(text)
        pdf.close()
        return "\n".join(pages)


class ContentParserContext:
    """Context that uses a ContentParserStrategy to parse content."""
    def __init__(self, strategy: ContentParserStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: ContentParserStrategy):
        self._strategy = strategy

    def parse(self, content: Any, source_url: str = "") -> Any:
        return self._strategy.parse(content, source_url)


# Shared instances for backward compatibility & context coordination
_table_strategy = TableParserStrategy()
_html_strategy = HtmlParserStrategy(table_parser=_table_strategy)
# Use Vision-based PDF parser (Docling) with PyMuPDF fallback
try:
    from app.pdf_vision_parser import VisionPdfParserStrategy
    _pdf_strategy = VisionPdfParserStrategy()
except ImportError:
    _pdf_strategy = PdfParserStrategy()


def _table_to_markdown(table_node) -> str:
    return _table_strategy.parse(table_node)


def _simhash(text: str) -> int:
    words = re.findall(r"\w+", text.lower())
    shingles = [" ".join(words[i:i+2]) for i in range(len(words)-1)]
    if not shingles:
        shingles = words if words else [text]
        
    v = [0] * 64
    for shingle in shingles:
        h = int(hashlib.sha1(shingle.encode("utf-8")).hexdigest()[:16], 16)
        for i in range(64):
            bit = (h >> i) & 1
            if bit:
                v[i] += 1
            else:
                v[i] -= 1
                
    fingerprint = 0
    for i in range(64):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def _extract_text_from_html(html: str, source_url: str = "") -> tuple[str, list[dict[str, str]]]:
    return _html_strategy.parse(html, source_url)


def _extract_text_from_pdf_bytes(payload: bytes) -> str:
    return _pdf_strategy.parse(payload)


def _extract_text_from_pdf_file(pdf_path: Path) -> str:
    return _pdf_strategy.parse(pdf_path)


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

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            is_separator_regex=False,
        )
        return splitter.split_text(text)
    except ImportError:
        # Fallback to naive string slicing if langchain is not installed
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


def _fetch_url(url: str, timeout: int = 20) -> tuple[str, str, list[dict[str, str]], str, str]:
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

    last_modified = response.headers.get("last-modified", "").strip()

    content_type = response.headers.get("content-type", "").lower()
    if url.lower().endswith(".pdf") or "application/pdf" in content_type:
        parser_context = ContentParserContext(_pdf_strategy)
        text = parser_context.parse(response.content)
        return text, "pdf", [], "", last_modified

    parser_context = ContentParserContext(_html_strategy)
    text, rejected = parser_context.parse(response.text, source_url=url)

    # If Last-Modified header was not present, look up meta tags in HTML
    if not last_modified and response.text:
        try:
            soup = BeautifulSoup(response.text, "html.parser")
            meta_selectors = [
                "meta[name='dcterms.modified']",
                "meta[property='article:modified_time']",
                "meta[name='last-modified']",
                "meta[name='date']"
            ]
            for selector in meta_selectors:
                meta = soup.select_one(selector)
                if meta and meta.get("content"):
                    last_modified = str(meta.get("content")).strip()
                    break
        except Exception:
            pass

    return text, "html", rejected, response.text, last_modified


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

    # Initialize processed_simhashes with existing records for near-deduplication
    processed_simhashes: dict[str, int] = {}
    for existing_url, chunk_list in existing_by_url.items():
        combined_text = "\n".join(c.get("text", "") for c in chunk_list)
        if combined_text:
            processed_simhashes[existing_url] = _simhash(combined_text)

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

        # Enforce domain-specific smart depth limit
        domain = urlparse(url).netloc.lower()
        is_gov = ".gov.au" in domain or ".edu.au" in domain
        max_depth = 2 if is_gov else 1
        if depth > max_depth:
            continue

        is_discovered_sub_link = depth > 0
        if is_discovered_sub_link:
            if url in discovered_sub_links_seen:
                continue
            if len(discovered_sub_links_tried) >= 30:  # Increased from 10 to 30 for deep harvesting
                discovery_exit_reason = "sub_links_exhausted"
                break
            discovered_sub_links_seen.add(url)
            discovered_sub_links_tried.append(url)
            time.sleep(random.uniform(2, 5))

        last_modified = ""
        try:
            text, extracted_type, rejected_from_html, raw_html, last_modified = _fetch_url(url)
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

        # Run SimHash near-deduplication check against already processed pages
        is_near_duplicate = False
        current_simhash = _simhash(text)
        for other_url, other_simhash in list(processed_simhashes.items()):
            if other_url == url:
                continue
            if _hamming_distance(current_simhash, other_simhash) <= 3:
                is_near_duplicate = True
                logger.info(f"Skipping {url} as it is a near-duplicate of {other_url}")
                if other_url in existing_by_url:
                    records.extend(existing_by_url[other_url])
                break

        if is_near_duplicate:
            continue

        processed_simhashes[url] = current_simhash

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

        # Use legal-aware chunking for legislation pages (preserves Section boundaries)
        legal_chunk_metadata: dict[int, dict] = {}
        if _is_legislation_url(url):
            legal_chunks = _chunk_legal_text(text, url)
            if legal_chunks:
                # Legal chunks include section metadata — extract text for quality check
                chunks = [lc["text"] for lc in legal_chunks]
                # Store metadata by index for later record creation
                for i, lc in enumerate(legal_chunks):
                    legal_chunk_metadata[i + 1] = {
                        "section_ref": lc.get("section_ref", ""),
                        "parent_context": lc.get("parent_context", ""),
                    }
            else:
                # No section structure found — fall back to standard chunking
                chunks = _chunk_text(text)
        else:
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
            # Tier 2: AI Content Evaluator (LLM scoring)
            ai_eval = _ai_evaluate_chunk(chunk)
            ai_score = ai_eval.get("score", 10)
            try:
                from app.config import settings as _cs
                ai_min = _cs.ai_evaluator_min_score
            except Exception:
                ai_min = 6
            if ai_score < ai_min:
                filtered_chunks_total += 1
                rejected_chunks.append(
                    {
                        "stage": "ai_evaluator",
                        "source_url": url,
                        "reason": f"ai_score={ai_score} < {ai_min}: {ai_eval.get('reason', '')}",
                        "text": chunk[:320],
                    }
                )
                continue
            hash_key = hashlib.sha1(f"{url}:{idx}:{chunk[:120]}".encode("utf-8")).hexdigest()[:16]
            # Get legal metadata from LegalChunker if available
            lc_meta = legal_chunk_metadata.get(idx, {})
            record = {
                "doc_id": f"doc_{hash_key}",
                "chunk_id": f"chunk_{hash_key}_{idx}",
                "source_url": url,
                "source_domain": urlparse(url).netloc.lower(),
                "source_type": source_type,
                "topic": _topic_from_url(url),
                "region": "AU",
                "title": title,
                "authority_score": 1.0 if _is_legislation_url(url) else (0.95 if "gov.au" in url else 0.8),
                "ai_relevance_score": ai_score,
                "approved": True,
                "text": chunk,
                # Legal metadata (Trụ Cột 2 + 3)
                "jurisdiction": _detect_jurisdiction(url),
                "act_name": _detect_act_name(url, chunk) if _is_legislation_url(url) else "",
                "section_ref": lc_meta.get("section_ref", ""),
                "parent_context": lc_meta.get("parent_context", ""),
                "last_modified": last_modified,
            }
            records.append(record)
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
                    parser_context = ContentParserContext(_pdf_strategy)
                    text = parser_context.parse(pdf_path)
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
