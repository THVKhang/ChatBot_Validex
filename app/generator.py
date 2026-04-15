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
    seed = quote_plus(f"{topic} {heading} editorial")
    return f"https://picsum.photos/seed/{seed}/1600/900"


def _clean_text(text: str, max_chars: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    clipped = compact[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{clipped}..." if clipped else compact[:max_chars]


def _looks_english(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False

    # Filter out Vietnamese-heavy snippets for user-facing English blog output.
    if re.search(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", candidate.lower()):
        return False

    words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]*", candidate)]
    if len(words) < 6:
        return False

    english_function_words = {
        "the", "and", "for", "with", "from", "that", "this", "are", "is", "to", "of", "in",
        "on", "as", "by", "or", "be", "can", "should", "will", "may", "at", "an", "a",
    }
    vietnamese_romanized_words = {
        "la", "quy", "trinh", "xac", "minh", "ly", "lich", "tu", "phap", "ung", "vien", "truoc", "khi",
        "nhan", "viec", "thuong", "ngay", "lam", "theo", "yeu", "cau", "anh", "huong", "den", "trong",
        "mot", "so", "cho", "ve", "duoc", "can", "noi", "bo", "dap", "ung", "quy", "dinh",
    }

    english_hits = sum(1 for item in words if item in english_function_words)
    vietnamese_hits = sum(1 for item in words if item in vietnamese_romanized_words)

    if vietnamese_hits >= 2 and english_hits <= 1:
        return False

    return english_hits >= 1


def _audience_label(audience: str) -> str:
    normalized = re.sub(r"\s+", " ", audience).strip().lower()
    if normalized in {"hr", "hr professionals"}:
        return "HR professionals"
    if normalized == "backoffice":
        return "backoffice teams"
    if not normalized:
        return "general audience"

    words: list[str] = []
    for token in normalized.split():
        if token == "hr":
            words.append("HR")
        else:
            words.append(token.capitalize())
    return " ".join(words)


def _is_police_check_topic(topic: str) -> bool:
    lowered = topic.lower()
    return "police check" in lowered or ("police" in lowered and "background" in lowered)


def _prefers_step_structure(parsed: ParsedPrompt) -> bool:
    prompt_lower = re.sub(r"\s+", " ", parsed.raw_prompt).strip().lower()

    # Respect explicit editorial/non-step requests.
    non_step_signals = [
        "not step",
        "non-step",
        "khong theo buoc",
        "không theo bước",
        "khong tung buoc",
        "không từng bước",
        "editorial",
        "thought leadership",
    ]
    if any(signal in prompt_lower for signal in non_step_signals):
        return False

    step_signals = [
        "step-by-step",
        "step by step",
        "how to",
        "checklist",
        "huong dan",
        "hướng dẫn",
        "guide",
        "walkthrough",
    ]
    return any(signal in prompt_lower for signal in step_signals)


def _join_bullets(items: list[str]) -> str:
    return "\n".join([f"- {item}" for item in items])


def extract_requested_image_limit(prompt: str) -> int | None:
    prompt_lower = re.sub(r"\s+", " ", prompt).strip().lower()

    image_terms = r"(?:image|images|picture|pictures|photo|photos|anh|ảnh|hinh|hình)"

    # No-image requests.
    if re.search(rf"\b(?:no|without)\s+(?:any\s+)?{image_terms}\b", prompt_lower):
        return 0
    if re.search(rf"\b(?:khong|không)\s+(?:co\s+)?{image_terms}\b", prompt_lower):
        return 0
    if re.search(rf"\b(?:remove|delete|drop)\s+(?:all\s+)?{image_terms}\b", prompt_lower):
        return 0
    if re.search(rf"\b(?:xoa|xóa|bo|bỏ)\s+(?:het\s+|toan\s+bo\s+|tat\s+ca\s+)?{image_terms}\b", prompt_lower):
        return 0

    number_words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "mot": 1,
        "một": 1,
    }

    match = re.search(
        rf"\b(?:keep|show|use|retain|limit(?:ed)?\s+to|only|chi|chỉ|giu|giữ)\s+(?:only\s+)?(\d+|zero|one|two|three|mot|một)\s+{image_terms}\b",
        prompt_lower,
    )
    if not match:
        match = re.search(rf"\b(\d+|zero|one|two|three|mot|một)\s+{image_terms}\b", prompt_lower)

    if not match:
        return None

    value = match.group(1)
    if value.isdigit():
        return max(0, int(value))
    return number_words.get(value)


def _section_plan(heading: str, topic: str, audience: str) -> tuple[str, str, list[str]]:
    heading_lower = heading.lower()

    if "understanding the national police check" in heading_lower:
        return (
            "A national police check should be treated as a risk-screening control, not a stand-alone compliance checkbox.",
            "For Australian employers, the practical objective is interpreting results with role context, legal boundaries, and fairness obligations in mind.",
            [
                "Define what the check confirms and what it does not confirm.",
                "Document role relevance criteria before assessing outcomes.",
                "Use consistent decision standards across hiring teams.",
            ],
        )

    if "role-based screening" in heading_lower or "roles requiring background verification" in heading_lower:
        return (
            "Screening depth should match the risk profile of each role and regulatory expectation.",
            "High-trust or vulnerable-sector roles usually justify stronger verification controls than low-risk operational roles.",
            [
                "Classify roles into risk tiers with clear rationale.",
                "Map each tier to mandatory checks and review thresholds.",
                "Review role-mapping quarterly with legal and compliance stakeholders.",
            ],
        )

    if "operational delivery" in heading_lower or "candidate experience" in heading_lower:
        return (
            "Execution quality is measured by both compliance accuracy and candidate experience.",
            "When communication is clear and timelines are realistic, employers reduce drop-off while maintaining governance standards.",
            [
                "Publish a transparent checklist before submission.",
                "Set turnaround expectations early and update candidates proactively.",
                "Track exception rates to identify process bottlenecks.",
            ],
        )

    if "employer responsibilities" in heading_lower or "compliance controls" in heading_lower:
        return (
            "Employer accountability extends beyond ordering checks to lawful handling, review discipline, and auditable decisions.",
            "A robust control model aligns recruitment operations with privacy, consent, retention, and policy governance obligations.",
            [
                "Maintain written procedures for consent, access, and retention.",
                "Restrict result visibility to authorized decision-makers only.",
                "Run periodic audits to validate policy adherence.",
            ],
        )

    if "next actions" in heading_lower:
        return (
            "The fastest improvement path is standardizing decisions before scaling volume.",
            "Teams that combine clear policy rules with operational metrics improve both hiring speed and compliance confidence.",
            [
                "Start with one role family and apply a consistent screening playbook.",
                "Measure turnaround, exception, and escalation trends monthly.",
                "Iterate policy and tooling based on observed risk patterns.",
            ],
        )

    if "core concepts" in heading_lower:
        return (
            f"For {audience}, separate police checks from broader background verification controls.",
            "Police checks focus on relevant criminal-history data, while broader verification confirms identity, eligibility, and role-risk fit across the hiring lifecycle.",
            [
                "Define which roles require police checks vs expanded verification.",
                "Map each control to policy, legal, or client contract requirements.",
                "Track exceptions in a central compliance register.",
            ],
        )

    if "practical" in heading_lower or "implementation" in heading_lower:
        return (
            f"Execution quality depends on a repeatable workflow that hiring and compliance teams can run consistently.",
            "Use clear ownership, turnaround SLAs, and escalation paths so screening does not become a bottleneck.",
            [
                "Publish a step-by-step checklist for recruiters and hiring managers.",
                "Collect consent and identity documents at the earliest stage.",
                "Communicate expected processing timelines to candidates proactively.",
            ],
        )

    if "conclusion" in heading_lower:
        return (
            f"A strong screening program protects business risk while preserving candidate trust.",
            f"The right approach for {topic} is to combine compliance rigor, transparent communication, and measurable operations.",
            [
                "Audit policy adherence monthly and review screening outcomes quarterly.",
                "Align recruitment KPIs with compliance KPIs, not just hiring speed.",
                "Define a clear owner for continuous process improvement.",
            ],
        )

    return (
        f"{topic.title()} should be treated as a strategic hiring-risk control, not only an administrative step.",
        f"For {audience}, the objective is balancing compliance, turnaround speed, and candidate experience.",
        [
            "Clarify screening scope by role sensitivity and regulatory expectation.",
            "Set measurable turnaround targets for each screening stage.",
            "Document decision rules to reduce inconsistent hiring outcomes.",
        ],
    )


def _supporting_facts(docs: list[RetrievedDoc], limit: int = 6) -> list[str]:
    if not docs:
        return []

    facts: list[str] = []
    for item in docs:
        snippet = _clean_text(item.content, 180)
        if not _looks_english(snippet):
            continue

        if snippet.lower().startswith("faq:"):
            snippet = snippet.split(":", 1)[1].strip()

        if snippet in facts:
            continue
        facts.append(snippet)
        if len(facts) >= limit:
            break

    return facts


def _section_body(parsed: ParsedPrompt, heading: str, evidence_fact: str | None) -> str:
    topic = re.sub(r"\s+", " ", parsed.topic).strip()
    audience = _audience_label(parsed.audience)
    heading_lower = heading.lower()
    is_police_topic = _is_police_check_topic(topic)

    evidence_block = ""
    if evidence_fact:
        evidence_block = f"Current guidance indicates: {evidence_fact}\n\n"

    if is_police_topic and heading_lower == "introduction":
        return (
            "Applying for a police check in Australia follows a defined, nationally coordinated process. "
            "While most applications are completed online, the same legal and identity verification requirements apply regardless of channel.\n\n"
            f"This guide outlines a practical path for {audience}, from preparing documents through to receiving results."
        )

    if is_police_topic and "overview of the application process" in heading_lower:
        process_steps = _join_bullets(
            [
                "Providing personal details",
                "Verifying identity using approved documents",
                "Submitting the application through an accredited provider",
                "Receiving the result once processing is complete",
            ]
        )
        expectations = _join_bullets(
            [
                "A fully online process in most cases",
                "Identity verification using documents and biometric comparison",
                "Variable processing times when additional review is required",
                "Secure digital delivery of results to the applicant",
            ]
        )
        return (
            "In Australia, police checks are typically name-based searches of criminal history records. "
            "Because checks rely on accurate identity matching, verification quality is essential.\n\n"
            "At a high level, the application process includes:\n"
            f"{process_steps}\n\n"
            "What to expect:\n"
            f"{expectations}\n\n"
            f"{evidence_block}".rstrip()
        ).strip()

    if is_police_topic and "step 1" in heading_lower:
        accepted_docs = _join_bullets(
            [
                "One commencement of identity document (birth/arrival basis)",
                "One primary community-use document",
                "One secondary community-use document",
                "At least one government-issued photo ID",
            ]
        )
        prep_tips = _join_bullets(
            [
                "Ensure documents are current and legible",
                "Use your full legal name consistently",
                "Check photo ID image quality before upload",
                "Use a device with a working camera for verification",
            ]
        )
        return (
            "Before you begin, gather your identity documents. Most police check workflows follow a three-category identity model.\n\n"
            "Accepted identity documents usually include:\n"
            f"{accepted_docs}\n\n"
            "Preparation tips:\n"
            f"{prep_tips}\n\n"
            f"{evidence_block}".rstrip()
        ).strip()

    if is_police_topic and "step 2" in heading_lower:
        form_items = _join_bullets(
            [
                "Full legal name and any previous names",
                "Date of birth and core identity details",
                "Address history (commonly up to five years)",
                "Purpose of the police check",
            ]
        )
        checklist = _join_bullets(
            [
                "All details match identity documents exactly",
                "Address history is complete",
                "Names are entered exactly as official records",
            ]
        )
        return (
            "Once documents are ready, complete the online application form. Accuracy at this stage prevents avoidable processing delays.\n\n"
            "Typical form fields include:\n"
            f"{form_items}\n\n"
            "Checklist before submitting:\n"
            f"{checklist}\n\n"
            f"{evidence_block}".rstrip()
        ).strip()

    if is_police_topic and "step 3" in heading_lower:
        verify_items = _join_bullets(
            [
                "Capture images of identity documents",
                "Validate document details against authoritative sources",
                "Complete a live selfie check",
            ]
        )
        return (
            "Identity verification is a mandatory control in Australian police check workflows. "
            "The goal is to ensure the applicant and identity claim match with high confidence.\n\n"
            "This step generally involves:\n"
            f"{verify_items}\n\n"
            "If digital verification cannot be completed, manual alternatives may be available, though processing often takes longer.\n\n"
            f"{evidence_block}".rstrip()
        ).strip()

    if is_police_topic and "step 4" in heading_lower:
        submit_items = _join_bullets(
            [
                "Submit through an accredited provider",
                "Complete payment at submission",
                "Enter national processing flow",
            ]
        )
        return (
            "After identity verification succeeds, submit the application and complete payment. "
            "Fees vary by check type and any applicable concession settings.\n\n"
            "At this stage:\n"
            f"{submit_items}\n\n"
            f"{evidence_block}".rstrip()
        ).strip()

    if is_police_topic and "step 5" in heading_lower:
        track_points = _join_bullets(
            [
                "Some checks are returned quickly",
                "Applications requiring additional assessment may take longer",
                "Final timelines are controlled by national processing once submitted",
            ]
        )
        return (
            "After submission, applicants can usually track progress online and receive results through a secure channel.\n\n"
            "Important timing notes:\n"
            f"{track_points}\n\n"
            "Police checks do not have a legislated expiry date; acceptance depends on the requesting organisation's policy.\n\n"
            f"{evidence_block}".rstrip()
        ).strip()

    if is_police_topic and "how validex supports" in heading_lower:
        validex_points = _join_bullets(
            [
                "Online application and identity verification",
                "Digital submission to nationally coordinated services",
                "Secure delivery of results",
            ]
        )
        return (
            "Validex supports end-to-end police check application workflows, including applicant consent, identity verification orchestration, and secure result delivery aligned with regulatory obligations.\n\n"
            "With Validex, applicants can complete:\n"
            f"{validex_points}"
        )

    if is_police_topic and "get started" in heading_lower:
        return (
            "Get started with your police check application today. "
            "Use a clear document checklist, complete identity verification carefully, and submit through an accredited channel for reliable processing."
        )

    lead, context, actions = _section_plan(heading, topic, audience)
    action_lines = _join_bullets(actions)
    blocks = [
        f"{lead}\n\n",
        f"{context}\n\n",
    ]
    if evidence_fact:
        blocks.append(f"A practical data point: {evidence_fact}\n\n")
    blocks.append("Recommended actions:\n")
    blocks.append(action_lines)
    return "".join(blocks)


def build_sections(parsed: ParsedPrompt, outline: list[str], docs: list[RetrievedDoc]) -> list[GeneratedBlog.Section]:
    supporting_facts = _supporting_facts(docs, limit=max(6, len(outline)))
    sections: list[GeneratedBlog.Section] = []
    for index, raw_heading in enumerate(outline):
        heading = raw_heading.split(":", 1)[-1].strip() if ":" in raw_heading else raw_heading.strip()
        if not heading:
            continue

        section_fact = supporting_facts[index] if index < len(supporting_facts) else None
        sections.append(
            GeneratedBlog.Section(
                heading=heading,
                body=_section_body(parsed, heading, section_fact),
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
    if _is_police_check_topic(parsed.topic):
        if _prefers_step_structure(parsed):
            if parsed.length == "short":
                return [
                    "Introduction",
                    "Overview of the Application Process",
                    "Step 1 - Gather Your Documents",
                    "Step 2 - Complete the Online Form",
                    "Get Started with Your Police Check",
                ]

            outline = [
                "Introduction",
                "Overview of the Application Process",
                "Step 1 - Gather Your Documents",
                "Step 2 - Complete the Online Form",
                "Step 3 - Verify Your Identity",
                "Step 4 - Submit and Pay",
                "Step 5 - Track Your Application and Receive Results",
                "How Validex Supports the Application Process",
                "Get Started with Your Police Check",
            ]

            if parsed.length == "long":
                outline.insert(2, "What to Expect")
            return outline

        if parsed.length == "short":
            return [
                "Introduction",
                "Understanding the National Police Check",
                "Risk, Compliance, and Candidate Experience",
                "Operational Priorities for Employers",
            ]

        outline = [
            "Introduction",
            "Understanding the National Police Check",
            "Role-Based Screening and Risk Governance",
            "Operational Delivery and Candidate Experience",
            "Employer Responsibilities and Compliance Controls",
            "Conclusion and Next Actions",
        ]

        if parsed.length == "long":
            outline.insert(4, "Building a Scalable Verification Program")
        return outline

    return [
        "Introduction",
        "Core concepts, context, and business value",
        "Practical implementation steps and compliance notes",
        "Conclusion and call to action",
    ]


def generate_draft(parsed: ParsedPrompt, docs: list[RetrievedDoc]) -> str:
    references = "\n".join([f"- {doc.doc_id} (score={doc.score})" for doc in docs])
    snippets = "\n".join([f"- {doc.content[:200]}" for doc in docs[:2]])

    if not references:
        references = "- No matching retrieval source was found"
    if not snippets:
        snippets = "- No reference snippets available"

    return (
        f"Suggested title: {parsed.topic.title()}\n\n"
        f"Tone: {parsed.tone}\n"
        f"Audience: {parsed.audience}\n"
        f"Length profile: {parsed.length}\n\n"
        "Starter direction:\n"
        f"This article explains {parsed.topic} with practical recommendations for operational teams. "
        "Add internal metrics and case examples where possible to strengthen credibility.\n\n"
        "Reference snippets:\n"
        f"{snippets}\n\n"
        "Retrieved sources:\n"
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

    previous_excerpt = ""
    if previous_draft:
        previous_excerpt = previous_draft[:350].strip()

    sections = build_sections(parsed, outline, docs)
    if previous_excerpt:
        sections.insert(
            0,
            GeneratedBlog.Section(
                heading="Context from Previous Draft",
                body=f"Related context carried from the previous turn:\n- {previous_excerpt}",
                image_url=build_section_image_url(parsed.topic, "previous draft context"),
                image_alt="Previous draft context",
            ),
        )

    if parsed.intent == "shorten":
        sections = sections[:3]
    if parsed.length == "long" and not _is_police_check_topic(parsed.topic):
        sections.extend(
            [
                GeneratedBlog.Section(
                    heading="Implementation Checklist",
                    body=(
                        f"Execution checklist for {parsed.audience}:\n"
                        "1) Define screening scope and policy boundaries.\n"
                        "2) Standardize required documents and consent steps.\n"
                        "3) Communicate timelines to candidates and hiring managers.\n"
                        "4) Track onboarding and compliance KPIs continuously."
                    ),
                    image_url=build_section_image_url(parsed.topic, "implementation checklist"),
                    image_alt="Implementation checklist",
                )
            ]
        )

    tone_line = parsed.tone.replace("_", " ")
    if not sections:
        sections = [
            GeneratedBlog.Section(
                heading="Overview",
                body=f"{clean_title} is presented in a {tone_line} tone for {parsed.audience}.",
                image_url=build_section_image_url(parsed.topic, "overview"),
                image_alt="Overview image",
            )
        ]

    draft = render_markdown_blog(clean_title, sections)
    if parsed.intent == "shorten":
        draft = f"> Shortened version\n\n{draft}"
    elif parsed.intent == "rewrite":
        draft = f"> Rewritten in the requested tone\n\n{draft}"

    return GeneratedBlog(
        title=clean_title,
        outline=outline,
        draft=draft,
        sources_used=sources_used,
        sections=sections,
    )


def generated_blog_to_dict(output: GeneratedBlog) -> dict:
    return asdict(output)
