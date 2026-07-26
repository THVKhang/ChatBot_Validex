"""Multi-Stage Writer Node — Plan → Draft → Self-Review pipeline.

All 3 stages are ACTIVE:
  Stage 1 (Plan): LLM generates structured outline with key points per section
  Stage 2 (Draft): LLM writes the full blog using outline + retrieved docs
  Stage 3 (Self-Review): LLM reviews and improves its own draft before Editor
"""
import json
import logging
import re
from langchain_core.documents import Document
from app.graph_state import GraphState
from app.langchain_pipeline import pipeline
from app.parser import ParsedPrompt, LANGUAGE_MAP
from app.generator import GeneratedBlog
from app.local_nli import verify_facts_nli

logger = logging.getLogger(__name__)


def _resolve_image_placeholders(draft: str, topic: str) -> str:
    """Replace ![IMAGE:keyword] placeholders AND fix fake/broken Unsplash URLs with real image URLs."""
    from app.generator import build_section_image_url

    # ── Pass 1: Resolve ![IMAGE:keyword] placeholders ──
    placeholder_pattern = re.compile(r'!\[IMAGE:([^\]]+)\]')
    matches = list(placeholder_pattern.finditer(draft))
    for match in reversed(matches):
        keyword = match.group(1).strip()
        image_url, alt_text = _fetch_real_image(keyword, topic)
        replacement = f"![{alt_text}]({image_url})"
        draft = draft[:match.start()] + replacement + draft[match.end():]
        logger.info(f"Writer: Resolved placeholder '{keyword}' → {image_url[:80]}...")

    # ── Pass 2: Fix fake/hallucinated Unsplash URLs ──
    # LLM sometimes ignores instructions and generates fake URLs like:
    # ![alt](https://images.unsplash.com/photo-XXXXX?w=800)
    fake_url_pattern = re.compile(
        r'!\[([^\]]*)\]\((https://images\.unsplash\.com/photo-[A-Z0-9X]{3,}[^\)]*)\)'
    )
    fake_matches = list(fake_url_pattern.finditer(draft))
    for match in reversed(fake_matches):
        alt = match.group(1).strip()
        fake_url = match.group(2)
        # Only fix if URL contains obvious placeholder patterns
        if 'XXXXX' in fake_url or 'photo-X' in fake_url or re.search(r'photo-[A-Z]{5,}', fake_url):
            keyword = alt or topic
            image_url, alt_text = _fetch_real_image(keyword, topic)
            replacement = f"![{alt_text}]({image_url})"
            draft = draft[:match.start()] + replacement + draft[match.end():]
            logger.info(f"Writer: Fixed fake URL for '{alt}' → {image_url[:80]}...")

    # ── Pass 3: Fix broken markdown image syntax ──
    # LLM sometimes outputs images without proper markdown syntax:
    #   "alt text(https://images.unsplash.com/...)" — missing ![...]
    #   "[alt text](https://images.unsplash.com/...)" — missing leading !
    # Fix: Convert to proper ![alt](url) format
    broken_img_pattern = re.compile(
        r'(?<!!)\[([^\]]+)\]\((https://images\.unsplash\.com/[^\)]+)\)'
    )
    broken_matches = list(broken_img_pattern.finditer(draft))
    for match in reversed(broken_matches):
        alt = match.group(1).strip()
        url = match.group(2)
        replacement = f"![{alt}]({url})"
        draft = draft[:match.start()] + replacement + draft[match.end():]
        logger.info(f"Writer: Fixed broken image link (missing !) for '{alt}'")

    # ── Pass 4: Fix raw "alt text(url)" without any brackets ──
    raw_img_pattern = re.compile(
        r'^([^!\[\n][^\(\n]{3,80})\((https://images\.unsplash\.com/[^\)]+)\)\s*$',
        re.MULTILINE
    )
    raw_matches = list(raw_img_pattern.finditer(draft))
    for match in reversed(raw_matches):
        alt = match.group(1).strip()
        url = match.group(2)
        replacement = f"![{alt}]({url})"
        draft = draft[:match.start()] + replacement + draft[match.end():]
        logger.info(f"Writer: Fixed raw image text for '{alt}'")

    return draft


def _fetch_real_image(keyword: str, topic: str) -> tuple[str, str]:
    """Fetch a real image URL from Unsplash API, fallback to Picsum."""
    from app.generator import build_section_image_url
    image_url = None
    alt_text = keyword

    try:
        result = pipeline._search_unsplash_image(keyword)
        if result and result[0]:
            image_url = result[0]
            alt_text = result[1] or keyword
    except Exception:
        pass

    if not image_url:
        image_url = build_section_image_url(topic, keyword)

    return (image_url, alt_text)


def _plan_outline(parsed: ParsedPrompt, docs: list[Document]) -> list[dict]:
    """Stage 1: LLM generates a detailed outline with key points per section."""
    # Use fast LLM for planning
    llm_to_use = getattr(pipeline, "_fast_llm", pipeline._llm) or pipeline._llm
    if llm_to_use is None:
        return []

    doc_summaries = "\n".join([
        f"- [{d.metadata.get('title', 'Source')}]: {d.page_content[:200]}"
        for d in docs[:5]
    ])

    # Determine section count from user settings
    target_sections = parsed.target_sections if parsed.target_sections > 0 else 5

    # Detect if user wants step-by-step format
    prompt_lower = parsed.raw_prompt.lower()
    is_howto = any(signal in prompt_lower for signal in [
        "step-by-step", "step by step", "how to", "how do i", "how can i",
        "guide", "walkthrough", "tutorial", "apply for", "applying for",
    ])

    if is_howto:
        format_instruction = (
            "CRITICAL FORMAT RULES:\n"
            "1. Structure the outline as NUMBERED STEPS (Step 1, Step 2, Step 3, etc.)\n"
            "2. Each step must be a concrete, actionable instruction the reader can follow\n"
            "3. Steps should be in chronological order of the actual process\n"
            "4. First section = Introduction, Last section = Conclusion\n"
            "5. DO NOT create generic headings like 'Key Factors' or 'What This Means for You'\n"
            "6. Example step headings: 'Step 1: Determine Which Type of Check You Need', "
            "'Step 2: Prepare Your 100-Point ID Documents', 'Step 3: Choose an Accredited Provider'\n"
        )
    else:
        format_instruction = (
            "CRITICAL FORMAT RULES:\n"
            "1. Make headings specific and compelling, not generic\n"
            "2. First section = Introduction, Last section = Conclusion and Strategic Next Steps\n"
        )

    prompt = (
        f"You are planning a blog article about: {parsed.topic}\n"
        f"Audience: {parsed.audience} | Tone: {parsed.tone} | Length: {parsed.length}\n"
        f"Language: Write in {LANGUAGE_MAP.get(parsed.language, 'English')}\n\n"
        "Available source material:\n"
        f"{doc_summaries}\n\n"
        "Create a detailed outline for this blog. Return ONLY a JSON array where each item has:\n"
        '- "heading": section heading\n'
        '- "key_points": array of 2-3 key points to cover\n'
        '- "relevant_sources": which sources to cite\n\n'
        f"{format_instruction}\n"
        f"Include {target_sections}-{target_sections + 2} sections to ensure the final blog exceeds 700 words.\n"
        f"TOPIC FOCUS: Stay strictly on '{parsed.topic}'. Do NOT mix in information about different types of checks "
        "unless the user explicitly asks for comparison. For example, if the topic is 'police check', do NOT "
        "include Working With Children Check (WWCC) details unless directly relevant.\n"
        "Each section must cover DIFFERENT aspects — do NOT repeat the same facts across sections.\n"
        "Return ONLY the JSON array."
    )

    try:
        response = llm_to_use.invoke(prompt)
        raw = getattr(response, "content", str(response))
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            outline = json.loads(match.group(0))
            if isinstance(outline, list) and len(outline) >= 3:
                logger.info(f"Writer Plan: generated outline with {len(outline)} sections")
                return outline
    except Exception as exc:
        logger.warning(f"Writer Plan failed: {exc}")

    return []


def _self_review(draft: str, parsed: ParsedPrompt, docs: list[Document]) -> str:
    """Stage 3: LLM self-reviews and improves the draft before Editor."""
    # Use main LLM (0.7 temperature) for review to maintain vocabulary flexibility and prose quality
    llm_to_use = getattr(pipeline, "_llm", None) or pipeline._llm
    if llm_to_use is None or not draft:
        return draft

    doc_titles = [d.metadata.get("title", "Source") for d in docs[:5]]

    # Detect if step-by-step was requested
    prompt_lower = parsed.raw_prompt.lower()
    is_howto = any(signal in prompt_lower for signal in [
        "step-by-step", "step by step", "how to", "guide", "apply for",
    ])

    format_check = ""
    if is_howto:
        format_check = (
            "6. FORMAT COMPLIANCE: The title promises a step-by-step guide. "
            "VERIFY the draft has numbered steps (Step 1, Step 2, etc.). "
            "If it does NOT, REWRITE the body sections as numbered steps. "
            "Each step should be a concrete, actionable instruction.\n"
        )

    prompt = (
        "You are a senior editorial reviewer. Review this blog draft and IMPROVE it.\n\n"
        "Check for:\n"
        "1. ACCURACY: Are all claims supported by the available sources? Remove unsupported claims. "
        "CRITICAL FACT: Australian National Police Checks (ACIC) do NOT have an expiry date. They are point-in-time checks.\n"
        "2. CITATIONS: Does every factual statement have a [Source: ...] citation?\n"
        "3. COHERENCE: Do sections flow logically? Are transitions smooth?\n"
        "4. COMPLETENESS: Are all key aspects of the topic covered?\n"
        "5. TONE: Is the tone consistent and appropriate for the target audience?\n"
        f"{format_check}"
        "7. NO REPETITION: If the SAME fact, statistic, or phrase appears in multiple sections, "
        "REMOVE the duplicates. Each section must present UNIQUE information. "
        "Common repetition: 'ACIC facilitates NPCS' or '5 million checks per year' — "
        "these should appear ONCE, not in every section.\n"
        "8. TOPIC FOCUS: Stay strictly on the main topic. "
        "Do NOT mix in unrelated check types. For example, if the topic is 'police check', "
        "do NOT include Working With Children Check (WWCC) details.\n"
        "9. NO FILE PATHS: If you see any 'file://C:/' or local disk paths, REMOVE them entirely.\n"
        "10. READABILITY: The blog MUST be readable at a Grade 8-10 level (Flesch-Kincaid). "
        "Replace long, complex sentences with short ones (max 20 words). "
        "Replace jargon with plain language. Break dense paragraphs into shorter ones. "
        "Use bullet points or numbered lists to simplify complex information.\n\n"
        f"Topic: {parsed.topic}\n"
        f"Audience: {parsed.audience}\n"
        f"Tone: {parsed.tone}\n"
        f"Available sources: {', '.join(doc_titles)}\n\n"
        f"Draft to review:\n{draft[:4000]}\n\n"
        "Return the IMPROVED version of the draft. Keep the same markdown format.\n"
        "Fix any issues you find. Do NOT add conversational commentary.\n"
        "Output ONLY the improved markdown blog post."
    )

    try:
        response = llm_to_use.invoke(prompt)
        improved = getattr(response, "content", str(response)).strip()
        # Basic validation
        if len(improved) > len(draft) * 0.5 and "##" in improved:
            logger.info(f"Writer Self-Review: improved draft ({len(draft)} → {len(improved)} chars)")
            return improved
        logger.warning("Self-review output failed validation, keeping original")
    except Exception as exc:
        logger.warning(f"Writer Self-Review failed: {exc}")

    return draft


def _inject_outline_into_instructions(parsed: ParsedPrompt, outline: list[dict]) -> ParsedPrompt:
    """Inject the planned outline into custom_instructions so the generator follows it."""
    if not outline:
        return parsed

    outline_text = (
        "MANDATORY OUTLINE — YOU MUST USE THESE EXACT HEADINGS IN ORDER.\n"
        "Each section MUST cover ONLY its assigned key points. Do NOT repeat information across sections.\n"
        "Do NOT add extra sections or change the heading text.\n\n"
    )
    for i, section in enumerate(outline, 1):
        heading = section.get("heading", f"Section {i}")
        key_points = section.get("key_points", [])
        points_str = "; ".join(key_points) if key_points else ""
        outline_text += f"{i}. {heading}"
        if points_str:
            outline_text += f" — Cover ONLY: {points_str}"
        outline_text += "\n"

    if parsed.custom_instructions:
        parsed.custom_instructions = f"{outline_text}\n{parsed.custom_instructions}"
    else:
        parsed.custom_instructions = outline_text

    return parsed


from app.agents.base import BaseAgentNode

class WriterAgentNode(BaseAgentNode):
    def execute(self, state: GraphState) -> GraphState:
        """Multi-stage writer: Plan → Draft → Self-Review."""
        from app.langchain_pipeline import pipeline
        revision_count = state.get("revision_count", 0)
        logger.info(f"Executing Smart Writer Node (Revision {revision_count})")
    
        parsed_dict = state["parsed"]
        parsed = ParsedPrompt(
            raw_prompt=state["prompt"],
            intent=parsed_dict["intent"],
            topic=parsed_dict["topic"],
            audience=parsed_dict["audience"],
            tone=parsed_dict["tone"],
            length=parsed_dict["length"],
            language=parsed_dict.get("language", "en"),
            target_sections=parsed_dict.get("target_sections", 0),
            target_images=parsed_dict.get("target_images", -1),
            custom_instructions=parsed_dict.get("context_note", "")
        )
    
        # Inject language instruction
        lang_name = LANGUAGE_MAP.get(parsed.language, "English")
        if parsed.language != "en":
            lang_instruction = f"IMPORTANT: Write the ENTIRE blog in {lang_name}. All headings, paragraphs, and conclusions must be in {lang_name}."
            if parsed.custom_instructions:
                parsed.custom_instructions = f"{lang_instruction}\n\n{parsed.custom_instructions}"
            else:
                parsed.custom_instructions = lang_instruction
    
        # Convert retrieved docs to Document objects
        retrieved_docs = state.get("retrieved_docs", [])
        docs = [
            Document(
                page_content=d["content"],
                metadata={
                    "doc_id": d["doc_id"],
                    "score": d["score"],
                    "source": d.get("source", ""),
                    "title": d.get("title", ""),
                    "source_url": d.get("source_url", ""),
                }
            )
            for d in retrieved_docs
        ]
    
        # Handle editor feedback (revision loop) — Reinforcement Learning
        feedback = state.get("editor_feedback")
        if feedback:
            logger.info(f"Writer incorporating editor feedback: {feedback}")
            
            # ── Targeted readability reinforcement ──
            # When editor detects R01 (FK Grade too high) or R02 (Flesch too low),
            # inject specific rewriting rules instead of vague "fix readability"
            readability_boost = ""
            if "R01:" in feedback or "R02:" in feedback:
                readability_boost = (
                    "\n\nREADABILITY REWRITE RULES (MANDATORY):\n"
                    "- Split ALL sentences longer than 20 words into 2 shorter sentences.\n"
                    "- Replace multi-syllable jargon with plain words "
                    "(e.g., 'utilise' → 'use', 'facilitate' → 'help', 'implementation' → 'setup').\n"
                    "- Convert dense paragraphs into bullet-point lists where possible.\n"
                    "- Use active voice only (e.g., 'The system checks...' NOT 'Checks are performed by...').\n"
                    "- Target Flesch-Kincaid Grade Level 8-10. Write as if explaining to a smart 14-year-old.\n"
                )
            
            # ── Repetition reinforcement ──
            repetition_boost = ""
            if "E01:repetition" in feedback:
                repetition_boost = (
                    "\n\nREPETITION FIX RULES (MANDATORY):\n"
                    "- Scan for any phrase repeated 3+ times across sections.\n"
                    "- Replace repeated phrases with synonyms or rephrase entirely.\n"
                    "- Each section must present unique information. Do NOT restate facts from other sections.\n"
                )
            
            revision_instruction = f"EDITOR FEEDBACK: {feedback}{readability_boost}{repetition_boost}"
            if parsed.custom_instructions:
                parsed.custom_instructions = f"{revision_instruction}\n\nORIGINAL INSTRUCTIONS: {parsed.custom_instructions}"
            else:
                parsed.custom_instructions = revision_instruction
    
        # Get previous draft for continuation mode
        previous_draft = None
        last_turn = state["session"].latest_turn()
        if last_turn and last_turn.generated_draft:
            previous_draft = last_turn.generated_draft
            logger.info(f"Continuation mode: using previous draft ({len(previous_draft)} chars)")
    
        if state.get("draft") and revision_count > 0:
            previous_draft = state["draft"]
    
        # ── FAST-PATH: Edit Existing Blog ──────────────────────────
        # When edit_instruction is set (e.g., "add 1 more picture", "make it shorter"),
        # apply the edit directly to the existing blog using LLM, skip full regeneration.
        edit_instruction = state.get("edit_instruction")
        if edit_instruction and previous_draft:
            logger.info(f"Writer: EDIT MODE — applying '{edit_instruction}' to existing draft")
            llm_to_use = getattr(pipeline, "_llm", None)
            if llm_to_use:
                # ── Detect if user wants a FORMAT CHANGE vs a CONTENT EDIT ──
                edit_lower = edit_instruction.lower()
                FORMAT_CHANGE_SIGNALS = [
                    "stop writing", "don't write", "don't use blog", "not a blog",
                    "instead of blog", "no longer a blog", "no more blog",
                    "convert to", "convert into", "write as", "rewrite as",
                    "format as", "change format", "change to",
                    "as an email", "as a memo", "as a letter", "as a report",
                    "as a summary", "as bullet points", "as json", "as plain text",
                    "internal email", "write an email", "write a memo",
                    "summarize into", "summarize as", "summarize the",
                    "just the table", "table only", "only the table",
                    "dừng viết blog", "viết dạng email", "viết thành email",
                    "chuyển thành", "đổi sang", "tóm tắt thành",
                ]
                is_format_change = any(signal in edit_lower for signal in FORMAT_CHANGE_SIGNALS)

                # ── Detect if previous_draft is a blog (has ## headings) or non-blog (email/table/etc) ──
                is_previous_blog = bool(re.search(r'^#{1,3}\s+', previous_draft, re.MULTILINE))

                if is_format_change:
                    logger.info(f"Writer: FORMAT CHANGE detected — allowing complete restructuring")
                    edit_prompt = (
                        "You are a content transformer. The user wants to COMPLETELY CHANGE "
                        "the format/structure of existing content.\n\n"
                        f"USER REQUEST: {edit_instruction}\n\n"
                        f"SOURCE CONTENT (use the facts and data from this, but change the format entirely):\n"
                        f"{previous_draft[:6000]}\n\n"
                        "CRITICAL RULES:\n"
                        "1. COMPLETELY CHANGE the output format to match the user's request.\n"
                        "2. If the user asks for an email, output ONLY an email. "
                        "Do NOT include blog headings like 'Introduction' or 'Conclusion'.\n"
                        "3. If the user asks for a table, output ONLY the table.\n"
                        "4. If the user asks for a summary, output ONLY a concise summary.\n"
                        "5. Use the FACTS and DATA from the source content, but restructure completely.\n"
                        "6. Do NOT wrap the output in a blog template. No 'Introduction', no 'Conclusion'.\n"
                        "7. Follow any word count, format, or persona instructions from the user.\n"
                        "8. Output ONLY the requested content. No commentary or explanation.\n"
                    )
                elif is_previous_blog:
                    # Previous draft IS a blog — use blog editor mode
                    edit_prompt = (
                        "You are a blog editor. The user wants to modify an existing blog post.\n\n"
                        f"USER REQUEST: {edit_instruction}\n\n"
                        f"EXISTING BLOG:\n{previous_draft[:6000]}\n\n"
                        "Apply the user's requested changes to the blog. Rules:\n"
                        "1. Keep all existing content that the user did NOT ask to change.\n"
                        "2. Only modify what the user explicitly asked for.\n"
                        "3. IMAGE RULES (CRITICAL — follow EXACTLY):\n"
                        "   a) For ADDING new images: insert a placeholder in this EXACT format:\n"
                        "      ![IMAGE:search keyword here]\n"
                        "      Add EXACTLY the number of images the user requested.\n"
                        "   b) For CHANGING/REPLACING an existing image: remove the old image line "
                        "and insert a new placeholder in the same position:\n"
                        "      ![IMAGE:search keyword describing the new image]\n"
                        "   c) For REMOVING an image: simply delete that image line.\n"
                        "   d) Do NOT use any URL (no https://...). ONLY use ![IMAGE:keyword] placeholders.\n"
                        "   e) Do NOT output raw URLs or links to images. Always use the placeholder format.\n"
                        "   f) Keep all OTHER existing images (that the user did NOT ask to change) as-is.\n"
                        "4. Maintain the same markdown format, heading structure, and tone.\n"
                        "5. Output ONLY the modified blog post. No commentary or explanation.\n"
                    )
                else:
                    # Previous draft is NOT a blog (email, table, plain text, etc.)
                    # Use adaptive content editor — preserve the existing format, don't force blog structure
                    logger.info(f"Writer: NON-BLOG content detected — using adaptive content editor")
                    edit_prompt = (
                        "You are a content editor. The user wants to modify existing content.\n\n"
                        f"USER REQUEST: {edit_instruction}\n\n"
                        f"EXISTING CONTENT:\n{previous_draft[:6000]}\n\n"
                        "Apply the user's requested changes. Rules:\n"
                        "1. Keep the SAME format as the existing content (if it's an email, keep it as email; "
                        "if it's a table, keep it as table).\n"
                        "2. Apply the user's changes (e.g., make longer, make shorter, add details).\n"
                        "3. Do NOT convert into a blog post. Do NOT add headings like 'Introduction' or 'Conclusion'.\n"
                        "4. Do NOT add blog structure. Preserve the original content type and format.\n"
                        "5. Follow any word count instructions from the user.\n"
                        "6. Output ONLY the modified content. No commentary or explanation.\n"
                    )
                try:
                    response = llm_to_use.invoke(edit_prompt)
                    edited_draft = getattr(response, "content", str(response)).strip()
                    # Validate the edit produced something reasonable
                    # Adaptive validation based on content type
                    if is_format_change or not is_previous_blog:
                        # Non-blog content: just check it has meaningful length
                        is_valid = len(edited_draft) > 50
                    else:
                        # Blog content: check for heading structure
                        is_valid = len(edited_draft) > len(previous_draft) * 0.3 and ("##" in edited_draft or "#" in edited_draft)
                    if is_valid:
                        # ── Resolve image placeholders to real URLs ──
                        edited_draft = _resolve_image_placeholders(edited_draft, parsed.topic)
                        logger.info(f"Writer: Edit applied successfully ({len(previous_draft)} → {len(edited_draft)} chars)")
                        # Extract title from the edited draft
                        import re as _re
                        title_match = _re.search(r"^#\s+(.+)$", edited_draft, _re.MULTILINE)
                        title = title_match.group(1).strip() if title_match else state.get("title", parsed.topic)
                        return {
                            "title": title,
                            "outline": state.get("outline", []),
                            "draft": edited_draft,
                            "sources_used": state.get("sources_used", []),
                            "previous_draft": previous_draft,
                            "loop_step": state.get("loop_step", 0) + 1,
                        }
                    else:
                        # ── CRITICAL FIX: Do NOT fall through to full blog regeneration ──
                        # Fallthrough was the root cause of "write in 400 word" generating
                        # a completely new blog instead of expanding the email.
                        logger.warning("Writer: Edit output failed validation — returning previous draft unchanged")
                        return {
                            "title": state.get("title", parsed.topic),
                            "outline": state.get("outline", []),
                            "draft": previous_draft,
                            "sources_used": state.get("sources_used", []),
                            "previous_draft": previous_draft,
                            "loop_step": state.get("loop_step", 0) + 1,
                        }
                except Exception as exc:
                    # ── CRITICAL FIX: Do NOT fall through on exception either ──
                    logger.warning(f"Writer: Edit fast-path failed: {exc} — returning previous draft unchanged")
                    return {
                        "title": state.get("title", parsed.topic),
                        "outline": state.get("outline", []),
                        "draft": previous_draft,
                        "sources_used": state.get("sources_used", []),
                        "previous_draft": previous_draft,
                        "loop_step": state.get("loop_step", 0) + 1,
                    }
    
        # ── Stage 1: PLAN (only on first attempt, skip on revisions) ──
        if revision_count == 0 and not feedback:
            outline = _plan_outline(parsed, docs)
            if outline:
                parsed = _inject_outline_into_instructions(parsed, outline)
                logger.info(f"Writer Stage 1: Outline injected ({len(outline)} sections)")
            else:
                logger.info("Writer Stage 1: Outline skipped (LLM unavailable or failed)")
        else:
            logger.info("Writer Stage 1: Outline skipped (revision/feedback mode)")
    
        # ── Stage 2: DRAFT (Active RAG enabled) ──
        llm_trace = {}
        payload = {
            "effective_parsed": parsed,
            "previous_draft": previous_draft,
            "documents": docs,
            "llm_trace": llm_trace,
        }
        generated: GeneratedBlog | None = pipeline._generate(payload)
    
        is_fallback = getattr(pipeline, "_last_generation_mode", None) == "fallback"
        from_api = state.get("from_api", False)
    
        if not generated or (from_api and is_fallback):
            logger.error("LLM Generation failed completely. Pipeline must not bypass the LLM.")
            raise RuntimeError("LLM Pipeline failed. Generation engine is completely bypassing the LLM.")
    
        # ── Stage 3: SELF-REVIEW (only on first attempt, skip on revisions) ──
        final_draft = generated.draft
        if revision_count == 0 and not feedback:
            final_draft = _self_review(final_draft, parsed, docs)
            logger.info("Writer Stage 3: Self-review complete")
            
            # NLI Fact-Check to remove hallucinations
            context = "\n\n".join([d.page_content for d in docs])
            final_draft = verify_facts_nli(final_draft, context)
        else:
            logger.info("Writer Stage 3: Self-review skipped (revision mode)")
    
        loop_step = state.get("loop_step", 0) + 1
    
        # Hallucination fallback check when no docs are retrieved or fallback was used
        ret_status = state.get("retrieval_meta", {}).get("status", "ok")
        if (not retrieved_docs or ret_status in {"low_confidence", "out_of_domain", "no_match"}) and not edit_instruction:
            from app.langchain_pipeline import pipeline
            settings = pipeline.settings
            if not settings.allow_hybrid_fallback:
                return {
                    "title": "Need More Context",
                    "outline": [
                        "Refine your topic to match the available dataset",
                        "Add relevant documents to the knowledge base",
                        "Re-run ingestion to update processed data and metadata",
                    ],
                    "draft": "This query is outside the scope of the current knowledge base. (Hybrid fallback is disabled)",
                    "sources_used": [],
                    "previous_draft": state.get("draft", ""),
                    "loop_step": loop_step,
                }
            else:
                warning_text = settings.hybrid_warning_text
                if warning_text and warning_text not in final_draft:
                    final_draft = f"> [!WARNING]\n> {warning_text}\n\n{final_draft}"
    
        return {
            "title": generated.title,
            "outline": generated.outline,
            "draft": final_draft,
            "sources_used": generated.sources_used if ret_status not in {"low_confidence", "out_of_domain", "no_match"} else [],
            "previous_draft": state.get("draft", ""),  # Comparison Gate: store draft before revision
            "loop_step": loop_step,
            "global_step_count": state.get("global_step_count", 0) + 1,
        }

writer_node = WriterAgentNode()

