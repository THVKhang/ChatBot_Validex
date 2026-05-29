"""Hierarchical Legal Chunker — Splits Australian legislation by legal structure.

Instead of naive RecursiveCharacterTextSplitter (cuts mid-sentence/mid-section),
this chunker follows the AST (Abstract Syntax Tree) of Australian legislation:

    Act → Part → Division → Section → Subsection

━━━ KEY INNOVATION: Context Enrichment ("Gia phả") ━━━━━━━━━━━━━━━━━━━━━━
Every chunk automatically inherits its parent breadcrumb, so the LLM and
embedding model ALWAYS know the full legal context:

    [Crimes Act 1914 (Cth) - Part VIIC - Section 85ZM - Subsection (2)]
    The waiting period for adult convictions is 10 years...

This eliminates the #1 RAG failure mode: "Which section does this belong to?"
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Maximum chunk size (characters). Optimized for embedding models.
MAX_CHUNK_CHARS = 2500
# Minimum chunk size — skip trivially small fragments
MIN_CHUNK_CHARS = 80
# Overlap for sub-chunking when a section is too large
SUB_CHUNK_OVERLAP = 200


@dataclass
class LegalNode:
    """A node in the legislation AST (Act → Part → Division → Section)."""
    level: str              # 'act', 'part', 'division', 'section', 'subsection'
    number: str             # '85ZM', 'VIIC', '3', etc.
    title: str              # 'Spent convictions scheme'
    text: str               # Full text of this node
    children: list["LegalNode"] = field(default_factory=list)
    start_pos: int = 0      # Character position in original text

    @property
    def heading(self) -> str:
        """Human-readable heading for this node."""
        label = self.level.capitalize()
        if self.number:
            return f"{label} {self.number}" + (f" — {self.title}" if self.title else "")
        return self.title or label


# ── Regex Patterns for Australian Legislation ──────────────────────

# Part headings: "Part VIIC — Spent Convictions" or "Part 3—Rehabilitation"
_PART_PATTERN = re.compile(
    r'^\s*(?:Part\s+)([IVXLC]+[A-Z]*|\d+[A-Z]*)\s*(?:[-—–]\s*)?(.*)$',
    re.MULTILINE | re.IGNORECASE
)

# Division headings: "Division 3 — Exclusions" or "Division 2—Interpretation"  
_DIVISION_PATTERN = re.compile(
    r'^\s*(?:Division\s+)(\d+[A-Z]*)\s*(?:[-—–]\s*)?(.*)$',
    re.MULTILINE | re.IGNORECASE
)

# Section headings (Commonwealth style): "85ZM  Spent convictions scheme"
# Also handles: "Section 85ZM" or numbered "13  Meaning of..."
_SECTION_PATTERN = re.compile(
    r'^\s*(?:(?:Section|s\.?|Sec\.?)\s+)?'      # Optional "Section" prefix
    r'(\d+[A-Z]*(?:\.\d+)*)'                     # Section number: 85ZM, 13, 5.1
    r'\s{2,}'                                     # At least 2 spaces (separator)
    r'(\S.+?)$',                                  # Title (non-empty)
    re.MULTILINE
)

# Subsection: "(1)", "(2)(a)", "(a)", "(i)"
_SUBSECTION_PATTERN = re.compile(
    r'^\s*(\(\d+\)(?:\([a-z]\))?)\s+',
    re.MULTILINE
)

# Schedule headings: "Schedule 1—..."
_SCHEDULE_PATTERN = re.compile(
    r'^\s*(?:Schedule\s+)(\d+[A-Z]*)\s*(?:[-—–]\s*)?(.*)$',
    re.MULTILINE | re.IGNORECASE
)


@dataclass
class LegalChunk:
    """A chunk ready for embedding, with full genealogy context."""
    text: str                   # The chunk text WITH breadcrumb prefix
    raw_text: str               # The raw text WITHOUT breadcrumb
    breadcrumb: str             # "Crimes Act 1914 (Cth) > Part VIIC > Section 85ZM"
    section_ref: str            # "Part VIIC - Section 85ZM"
    act_name: str
    jurisdiction: str
    section_number: str
    section_title: str
    chunk_index: int            # Sub-chunk index (0 if no sub-chunking needed)
    metadata: dict[str, Any] = field(default_factory=dict)


class LegalChunker:
    """Hierarchical Legal Chunker for Australian Legislation.

    Usage::

        chunker = LegalChunker()
        chunks = chunker.chunk(
            text="Part VIIC — Spent Convictions\\n85ZM  Spent convictions scheme\\n...",
            act_name="Crimes Act 1914 (Cth)",
            jurisdiction="Commonwealth",
        )
        for chunk in chunks:
            print(chunk.breadcrumb)
            print(chunk.text[:200])
    """

    def __init__(
        self,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
        sub_chunk_overlap: int = SUB_CHUNK_OVERLAP,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.sub_chunk_overlap = sub_chunk_overlap

    def chunk(
        self,
        text: str,
        act_name: str = "",
        jurisdiction: str = "Commonwealth",
        source_url: str = "",
    ) -> list[LegalChunk]:
        """Split legal text into hierarchical chunks with context enrichment.

        Parameters
        ----------
        text : str
            The full text of the legislation (or a large section of it).
        act_name : str
            Name of the Act (e.g. 'Crimes Act 1914 (Cth)').
        jurisdiction : str
            Jurisdiction code (e.g. 'Commonwealth', 'NSW', 'VIC').
        source_url : str
            Source URL for metadata.

        Returns
        -------
        list[LegalChunk]
            Chunks with breadcrumb context, ready for embedding.
        """
        # Step 1: Find structural boundaries
        boundaries = self._find_boundaries(text)

        if not boundaries:
            # No legal structure detected — return text as-is with context
            return self._fallback_chunk(text, act_name, jurisdiction, source_url)

        # Step 2: Split text at boundaries
        raw_chunks = self._split_at_boundaries(text, boundaries)

        # Step 3: Build chunks with breadcrumb context enrichment
        result: list[LegalChunk] = []
        current_part = ""
        current_division = ""

        for raw in raw_chunks:
            level = raw["level"]
            number = raw["number"]
            title = raw["title"]
            chunk_text = raw["text"].strip()

            # Track current Part/Division for breadcrumb
            if level == "part":
                current_part = f"Part {number}" + (f" — {title}" if title else "")
                current_division = ""
            elif level == "division":
                current_division = f"Division {number}" + (f" — {title}" if title else "")
            elif level == "schedule":
                current_part = f"Schedule {number}" + (f" — {title}" if title else "")
                current_division = ""

            # Build breadcrumb (genealogy)
            breadcrumb_parts = [p for p in [act_name, current_part, current_division] if p]
            if level == "section":
                section_label = f"Section {number}" + (f" — {title}" if title else "")
                breadcrumb_parts.append(section_label)
                section_ref = " - ".join([p for p in [current_part, section_label] if p])
            else:
                section_ref = " - ".join(breadcrumb_parts[1:]) if len(breadcrumb_parts) > 1 else ""

            breadcrumb = " > ".join(breadcrumb_parts)

            # Skip trivially small chunks
            if len(chunk_text) < self.min_chunk_chars:
                continue

            # Sub-chunk if too large
            if len(chunk_text) > self.max_chunk_chars:
                sub_chunks = self._sub_chunk(chunk_text)
                for i, sub_text in enumerate(sub_chunks):
                    # Prepend breadcrumb to every sub-chunk
                    enriched_text = f"[{breadcrumb}]\n{sub_text}"
                    result.append(LegalChunk(
                        text=enriched_text,
                        raw_text=sub_text,
                        breadcrumb=breadcrumb,
                        section_ref=section_ref,
                        act_name=act_name,
                        jurisdiction=jurisdiction,
                        section_number=number,
                        section_title=title,
                        chunk_index=i,
                        metadata={"source_url": source_url, "level": level},
                    ))
            else:
                # Prepend breadcrumb
                enriched_text = f"[{breadcrumb}]\n{chunk_text}"
                result.append(LegalChunk(
                    text=enriched_text,
                    raw_text=chunk_text,
                    breadcrumb=breadcrumb,
                    section_ref=section_ref,
                    act_name=act_name,
                    jurisdiction=jurisdiction,
                    section_number=number,
                    section_title=title,
                    chunk_index=0,
                    metadata={"source_url": source_url, "level": level},
                ))

        logger.info(
            "LegalChunker: %s → %d chunks (act=%s, jurisdiction=%s)",
            source_url[:50] if source_url else "unknown",
            len(result), act_name, jurisdiction,
        )
        return result

    def _find_boundaries(self, text: str) -> list[dict]:
        """Find all structural boundaries in the text.

        Returns a sorted list of {pos, level, number, title} dicts.
        """
        boundaries: list[dict] = []

        for match in _PART_PATTERN.finditer(text):
            boundaries.append({
                "pos": match.start(),
                "level": "part",
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
            })

        for match in _DIVISION_PATTERN.finditer(text):
            boundaries.append({
                "pos": match.start(),
                "level": "division",
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
            })

        for match in _SECTION_PATTERN.finditer(text):
            boundaries.append({
                "pos": match.start(),
                "level": "section",
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
            })

        for match in _SCHEDULE_PATTERN.finditer(text):
            boundaries.append({
                "pos": match.start(),
                "level": "schedule",
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
            })

        # Sort by position
        boundaries.sort(key=lambda b: b["pos"])
        return boundaries

    def _split_at_boundaries(
        self, text: str, boundaries: list[dict]
    ) -> list[dict]:
        """Split text at structural boundaries, returning raw chunks."""
        chunks: list[dict] = []

        for i, boundary in enumerate(boundaries):
            start = boundary["pos"]
            end = boundaries[i + 1]["pos"] if i + 1 < len(boundaries) else len(text)
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "level": boundary["level"],
                    "number": boundary["number"],
                    "title": boundary["title"],
                    "text": chunk_text,
                })

        return chunks

    def _sub_chunk(self, text: str) -> list[str]:
        """Sub-chunk a large section by paragraph boundaries.

        Preserves paragraph integrity. Falls back to sentence boundaries
        if paragraphs are also too large.
        """
        # Try splitting by double newline (paragraph)
        paragraphs = re.split(r'\n\s*\n', text)

        if len(paragraphs) <= 1:
            # No paragraph breaks — split by sentences
            paragraphs = re.split(r'(?<=[.!?])\s+', text)

        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 <= self.max_chunk_chars:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        # If a single chunk is still too large, force-split at character level
        final_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) > self.max_chunk_chars:
                # Force-split with overlap
                start = 0
                while start < len(chunk):
                    end = min(len(chunk), start + self.max_chunk_chars)
                    final_chunks.append(chunk[start:end].strip())
                    if end >= len(chunk):
                        break
                    start = max(0, end - self.sub_chunk_overlap)
            else:
                final_chunks.append(chunk)

        return final_chunks if final_chunks else [text]

    def _fallback_chunk(
        self,
        text: str,
        act_name: str,
        jurisdiction: str,
        source_url: str,
    ) -> list[LegalChunk]:
        """Fallback for text without detectable legal structure.

        Still applies context enrichment with whatever we know.
        """
        breadcrumb = act_name if act_name else "Unknown Source"

        if len(text) <= self.max_chunk_chars:
            enriched = f"[{breadcrumb}]\n{text}"
            return [LegalChunk(
                text=enriched,
                raw_text=text,
                breadcrumb=breadcrumb,
                section_ref="",
                act_name=act_name,
                jurisdiction=jurisdiction,
                section_number="",
                section_title="",
                chunk_index=0,
                metadata={"source_url": source_url, "level": "unknown"},
            )]

        sub_chunks = self._sub_chunk(text)
        result = []
        for i, sub in enumerate(sub_chunks):
            enriched = f"[{breadcrumb}]\n{sub}"
            result.append(LegalChunk(
                text=enriched,
                raw_text=sub,
                breadcrumb=breadcrumb,
                section_ref="",
                act_name=act_name,
                jurisdiction=jurisdiction,
                section_number="",
                section_title="",
                chunk_index=i,
                metadata={"source_url": source_url, "level": "unknown"},
            ))
        return result


# ── Module-level convenience ─────────────────────────────────────

_DEFAULT_CHUNKER: LegalChunker | None = None


def get_legal_chunker() -> LegalChunker:
    """Get or create the default LegalChunker singleton."""
    global _DEFAULT_CHUNKER
    if _DEFAULT_CHUNKER is None:
        _DEFAULT_CHUNKER = LegalChunker()
    return _DEFAULT_CHUNKER


def chunk_legal_text(
    text: str,
    act_name: str = "",
    jurisdiction: str = "Commonwealth",
    source_url: str = "",
) -> list[LegalChunk]:
    """Convenience function — chunk legal text using the default chunker."""
    return get_legal_chunker().chunk(text, act_name, jurisdiction, source_url)
