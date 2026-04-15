from __future__ import annotations

import html
import re
from typing import Any


def _strip_markdown(text: str) -> str:
    normalized = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    normalized = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", normalized)
    normalized = re.sub(r"[`*_>#-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _meta_description(report: dict[str, Any], max_chars: int = 160) -> str:
    candidates: list[str] = []

    title = str(report.get("title", "")).strip()
    if title:
        candidates.append(title)

    draft = str(report.get("draft", "")).strip()
    if draft:
        candidates.append(_strip_markdown(draft))

    sections = report.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            body = str(section.get("body", "")).strip()
            if body:
                candidates.append(_strip_markdown(body))
                break

    for text in candidates:
        if len(text) >= 60:
            return text[:max_chars].rstrip(" .")

    fallback = (title or "Validex approved editorial draft").strip()
    return fallback[:max_chars].rstrip(" .")


def _normalize_alt(section: dict[str, Any], index: int) -> str:
    alt = str(section.get("image_alt", "")).strip()
    if alt:
        return alt

    heading = str(section.get("heading", "")).strip()
    if heading:
        return heading

    return f"Illustration {index}"


def _section_paragraphs(body: str) -> list[str]:
    lines = [line.strip() for line in re.split(r"\n+", body) if line.strip()]
    return lines or [""]


def build_publish_markdown(report: dict[str, Any]) -> str:
    title = str(report.get("title", "")).strip() or "Untitled"
    description = _meta_description(report)
    sections = report.get("sections")

    output: list[str] = [
        "---",
        f"title: {title}",
        f"meta_description: {description}",
        "---",
        "",
        f"# {title}",
        "",
    ]

    if isinstance(sections, list) and sections:
        for index, section in enumerate(sections, start=1):
            if not isinstance(section, dict):
                continue

            heading = str(section.get("heading", "")).strip() or f"Section {index}"
            body = str(section.get("body", "")).strip()
            image_url = str(section.get("image_url", "")).strip()
            image_alt = _normalize_alt(section, index)

            output.extend([f"## {heading}", ""])
            if image_url:
                output.extend([f"![{image_alt}]({image_url})", ""])

            if body:
                output.extend([body, ""])
    else:
        draft = str(report.get("draft", "")).strip()
        if draft:
            output.extend([draft, ""])

    sources = report.get("sources_used")
    if isinstance(sources, list) and sources:
        output.extend(["## Sources", ""])
        for source in sources:
            source_text = str(source).strip()
            if source_text:
                output.append(f"- {source_text}")

    return "\n".join(output).strip() + "\n"


def build_publish_html(report: dict[str, Any]) -> str:
    title = str(report.get("title", "")).strip() or "Untitled"
    escaped_title = html.escape(title)
    description = _meta_description(report)
    escaped_description = html.escape(description)
    sections = report.get("sections")

    body_sections: list[str] = []
    if isinstance(sections, list) and sections:
        for index, section in enumerate(sections, start=1):
            if not isinstance(section, dict):
                continue

            heading = str(section.get("heading", "")).strip() or f"Section {index}"
            body = str(section.get("body", "")).strip()
            image_url = str(section.get("image_url", "")).strip()
            image_alt = _normalize_alt(section, index)

            section_blocks: list[str] = [f"<section><h2>{html.escape(heading)}</h2>"]
            if image_url:
                section_blocks.append(
                    f"<figure><img src=\"{html.escape(image_url)}\" alt=\"{html.escape(image_alt)}\" loading=\"lazy\" /></figure>"
                )

            for paragraph in _section_paragraphs(body):
                if paragraph:
                    section_blocks.append(f"<p>{html.escape(paragraph)}</p>")

            section_blocks.append("</section>")
            body_sections.append("".join(section_blocks))
    else:
        draft = str(report.get("draft", "")).strip()
        if draft:
            body_sections.append(f"<section><p>{html.escape(_strip_markdown(draft))}</p></section>")

    sources_html = ""
    sources = report.get("sources_used")
    if isinstance(sources, list) and sources:
        items = "".join(
            f"<li>{html.escape(str(source).strip())}</li>"
            for source in sources
            if str(source).strip()
        )
        if items:
            sources_html = f"<section><h2>Sources</h2><ul>{items}</ul></section>"

    article_content = "".join(body_sections) + sources_html

    return (
        "<!doctype html>"
        "<html lang=\"en\">"
        "<head>"
        "<meta charset=\"utf-8\" />"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />"
        f"<title>{escaped_title}</title>"
        f"<meta name=\"description\" content=\"{escaped_description}\" />"
        "</head>"
        f"<body><article><h1>{escaped_title}</h1>{article_content}</article></body>"
        "</html>"
    )


def build_publish_output(report: dict[str, Any], output_format: str = "markdown") -> dict[str, str]:
    normalized = output_format.strip().lower()
    if normalized == "markdown":
        return {
            "format": "markdown",
            "mime_type": "text/markdown",
            "content": build_publish_markdown(report),
        }
    if normalized == "html":
        return {
            "format": "html",
            "mime_type": "text/html",
            "content": build_publish_html(report),
        }

    raise ValueError("Unsupported format. Use markdown or html.")
