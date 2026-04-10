from dataclasses import asdict, dataclass, field
import re
from urllib.parse import quote_plus

from app.parser import ParsedPrompt
from app.retriever import RetrievedDoc


@dataclass
class GeneratedBlog:
    @dataclass
    class Section:
        heading: str
        body: str
        image_url: str
        image_alt: str

    title: str
    outline: list[str]
    draft: str
    sources_used: list[str]
    sections: list[Section] = field(default_factory=list)


def build_section_image_url(topic: str, heading: str) -> str:
    query = quote_plus(f"{topic} {heading} professional editorial")
    return f"https://source.unsplash.com/1600x900/?{query}"


def _section_body(parsed: ParsedPrompt, heading: str, snippets: str) -> str:
    tone_line = parsed.tone.replace("_", " ")
    return (
        f"Trong phan \"{heading}\", bai viet tap trung vao nhu cau cua {parsed.audience} "
        f"voi giong van {tone_line}. Noi dung can ro rang, co tinh hanh dong va de ap dung trong thuc te.\n\n"
        f"Diem grounding tu tai lieu:\n{snippets}"
    )


def build_sections(parsed: ParsedPrompt, outline: list[str], docs: list[RetrievedDoc]) -> list[GeneratedBlog.Section]:
    snippets = "\n".join([f"- [{doc.doc_id}] {doc.content[:160]}" for doc in docs[:2]])
    if not snippets:
        snippets = "- Chua co tai lieu tham chieu"

    sections: list[GeneratedBlog.Section] = []
    for raw_heading in outline:
        heading = raw_heading.split(":", 1)[-1].strip() if ":" in raw_heading else raw_heading.strip()
        if not heading:
            continue
        sections.append(
            GeneratedBlog.Section(
                heading=heading,
                body=_section_body(parsed, heading, snippets),
                image_url=build_section_image_url(parsed.topic, heading),
                image_alt=f"{heading} illustration",
            )
        )
    return sections


def render_markdown_blog(title: str, sections: list[GeneratedBlog.Section]) -> str:
    blocks: list[str] = [f"# {title}"]
    for section in sections:
        blocks.extend(
            [
                "",
                f"## {section.heading}",
                "",
                f"![{section.image_alt}]({section.image_url})",
                "",
                section.body,
            ]
        )
    return "\n".join(blocks)


def format_title(topic: str) -> str:
    compact = re.sub(r"\s+", " ", topic).strip(" .,!?:;\n\t")
    if not compact or compact.lower() == "current draft":
        return "Current Draft Update"

    words = compact.split()
    if len(words) > 14:
        words = words[:14]
    compact = " ".join(words)

    stop_words = {"a", "an", "the", "and", "or", "for", "to", "of", "in", "on", "with"}
    titled: list[str] = []
    for index, word in enumerate(compact.split()):
        if index > 0 and index < len(compact.split()) - 1 and word.lower() in stop_words:
            titled.append(word.lower())
        else:
            titled.append(word.capitalize())
    return " ".join(titled)


def generate_outline(parsed: ParsedPrompt) -> list[str]:
    return [
        f"Mo dau: {parsed.topic}",
        "Noi dung chinh: dinh nghia, boi canh, gia tri",
        "Huong dan thuc te va luu y",
        "Ket luan va call-to-action",
    ]


def generate_draft(parsed: ParsedPrompt, docs: list[RetrievedDoc]) -> str:
    references = "\n".join([f"- {doc.doc_id} (score={doc.score})" for doc in docs])
    snippets = "\n".join([f"- {doc.content[:200]}" for doc in docs[:2]])

    if not references:
        references = "- Khong tim thay tai lieu phu hop"
    if not snippets:
        snippets = "- Chua co snippet tham chieu"

    return (
        f"Tieu de goi y: {parsed.topic.title()}\n\n"
        f"Giong van: {parsed.tone}\n"
        f"Doi tuong doc gia: {parsed.audience}\n"
        f"Do dai: {parsed.length}\n\n"
        "Noi dung nhap:\n"
        f"Bai viet nay giai thich ve {parsed.topic}. "
        "Ban co the bo sung so lieu va case study noi bo de tang do tin cay.\n\n"
        "Snippet tham chieu:\n"
        f"{snippets}\n\n"
        "Nguon da truy xuat:\n"
        f"{references}\n"
    )


def generate_blog_output(
    parsed: ParsedPrompt,
    docs: list[RetrievedDoc],
    previous_draft: str | None = None,
) -> GeneratedBlog:
    outline = generate_outline(parsed)
    sources_used = [doc.doc_id for doc in docs]
    clean_title = format_title(parsed.topic)

    source_snippets = "\n".join([f"- {doc.content[:220]}" for doc in docs[:3]])
    if not source_snippets:
        source_snippets = "- No retrieval context available"

    grounded_points = "\n".join(
        [f"- [{doc.doc_id}] {doc.content[:140]}" for doc in docs[:3]]
    )
    if not grounded_points:
        grounded_points = "- Chua co diem du lieu de grounding"

    tone_line = parsed.tone.replace("_", " ")
    previous_excerpt = ""
    if previous_draft:
        previous_excerpt = previous_draft[:350].strip()

    sections = build_sections(parsed, outline, docs)
    if previous_excerpt:
        sections.insert(
            0,
            GeneratedBlog.Section(
                heading="Context from Previous Draft",
                body=f"Tom tat noi dung lien quan tu luot truoc:\n- {previous_excerpt}",
                image_url=build_section_image_url(parsed.topic, "previous draft context"),
                image_alt="Previous draft context",
            ),
        )

    if parsed.intent == "shorten":
        sections = sections[:3]
    if parsed.length == "long":
        sections.extend(
            [
                GeneratedBlog.Section(
                    heading="Implementation Checklist",
                    body=(
                        f"De trien khai cho {parsed.audience}:\n"
                        "1) Xac dinh pham vi kiem tra.\n"
                        "2) Chuan hoa tai lieu bat buoc.\n"
                        "3) Truyen thong timeline cho ung vien.\n"
                        "4) Theo doi KPI onboarding va compliance."
                    ),
                    image_url=build_section_image_url(parsed.topic, "implementation checklist"),
                    image_alt="Implementation checklist",
                )
            ]
        )

    if not sections:
        sections = [
            GeneratedBlog.Section(
                heading="Overview",
                body=f"{clean_title} duoc trinh bay theo tone {tone_line} cho {parsed.audience}.",
                image_url=build_section_image_url(parsed.topic, "overview"),
                image_alt="Overview image",
            )
        ]

    draft = render_markdown_blog(clean_title, sections)
    if parsed.intent == "shorten":
        draft = f"> Phien ban rut gon\n\n{draft}"
    elif parsed.intent == "rewrite":
        draft = f"> Rewrite theo tone yeu cau\n\n{draft}"

    return GeneratedBlog(
        title=clean_title,
        outline=outline,
        draft=draft,
        sources_used=sources_used,
        sections=sections,
    )


def generated_blog_to_dict(output: GeneratedBlog) -> dict:
    return asdict(output)
