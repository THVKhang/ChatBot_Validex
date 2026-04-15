from app import collect_au_sources


def test_quality_chunk_accepts_relevant_legal_content():
    text = (
        "Australia police check application guidance explains identity verification, "
        "background screening obligations, and legislation references for employment compliance "
        "teams handling recruitment reviews while documenting applicant consent controls, "
        "record retention expectations, and audit-ready governance notes."
    )

    assert collect_au_sources._is_quality_chunk(text, min_words=25, min_keyword_matches=2)


def test_quality_chunk_rejects_short_or_irrelevant_text():
    short_text = "Police check note only."
    generic_text = (
        "This paragraph talks about workplace culture improvements and communication plans "
        "for teams and managers without legal compliance context or source-specific details."
    )

    assert not collect_au_sources._is_quality_chunk(short_text, min_words=25, min_keyword_matches=2)
    assert not collect_au_sources._is_quality_chunk(generic_text, min_words=25, min_keyword_matches=2)


def test_extract_html_prefers_main_and_removes_noise_phrases():
    html = """
    <html>
        <body>
            <article>
                <p>Cookie Policy: accept all settings.</p>
            </article>
            <main>
                <p>AFP and ACIC guidance helps each applicant verify identity before submission. The result is reviewed for conviction relevance.</p>
                <p>Follow us on social channels for promotions.</p>
            </main>
        </body>
    </html>
    """

    text, rejected = collect_au_sources._extract_text_from_html(html, source_url="https://example.gov.au")

    lowered = text.lower()
    assert "afp and acic guidance" in lowered
    assert "cookie policy" not in lowered
    assert "follow us on" not in lowered
    assert any(item.get("stage") == "noise_phrase" for item in rejected)


def test_extract_html_semantic_filter_rejects_non_paragraph_noise():
    html = """
    <html>
        <body>
            <main>
                <p>***** ----- #####</p>
                <p>ABCDEF 12345 ZXCVBNM</p>
            </main>
        </body>
    </html>
    """

    text, rejected = collect_au_sources._extract_text_from_html(html, source_url="https://example.gov.au")

    assert text == ""
    assert any(item.get("stage") == "semantic_paragraph" for item in rejected)


def test_fetch_url_requires_stealth_when_flag_enabled(monkeypatch):
    monkeypatch.setattr(collect_au_sources, "COLLECT_REQUIRE_STEALTH", True)
    monkeypatch.setattr(collect_au_sources, "curl_requests", None)

    try:
        collect_au_sources._fetch_url("https://www.oaic.gov.au/privacy")
    except RuntimeError as exc:
        assert "COLLECT_REQUIRE_STEALTH=1" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when COLLECT_REQUIRE_STEALTH is enabled without curl_cffi")


def test_collect_sources_discovers_hub_sub_links_and_stops_at_target(monkeypatch, tmp_path):
    monkeypatch.setattr(collect_au_sources, "COLLECT_REQUIRE_STEALTH", False)
    monkeypatch.setattr(collect_au_sources.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(collect_au_sources.random, "uniform", lambda _a, _b: 2.5)

    hub_html = """
    <html>
        <body>
            <main>
                <a href="/guidance/topic-1">Guidance 1</a>
                <a href="/advice/topic-2">Advice 2</a>
                <a href="/decision/topic-3">Decision 3</a>
                <a href="/report/topic-4">Report 4</a>
                <a href="/guidance/topic-5">Guidance 5</a>
                <a href="/advice/topic-6">Advice 6</a>
                <a href="/decision/topic-7">Decision 7</a>
                <a href="/report/topic-8">Report 8</a>
                <a href="/guidance/topic-9">Guidance 9</a>
                <a href="/advice/topic-10">Advice 10</a>
                <a href="/report/topic-11">Ignored because of limit</a>
            </main>
        </body>
    </html>
    """

    sublink_text = (
        "AFP and ACIC verify applicant identity and conviction history for Australian police check compliance. "
        "This supports lawful screening decisions and result handling across employer workflows. "
        * 14
    )

    fetch_map = {
        "https://www.oaic.gov.au/privacy": (
            "OAIC privacy hub. See guidance and advice for official information.",
            "html",
            [],
            hub_html,
        ),
        "https://www.oaic.gov.au/guidance/topic-1": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/advice/topic-2": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/decision/topic-3": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/report/topic-4": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/guidance/topic-5": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/advice/topic-6": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/decision/topic-7": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/report/topic-8": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/guidance/topic-9": (sublink_text, "html", [], ""),
        "https://www.oaic.gov.au/advice/topic-10": (sublink_text, "html", [], ""),
    }

    def fake_fetch(url, timeout=20):
        return fetch_map[url]

    monkeypatch.setattr(collect_au_sources, "_fetch_url", fake_fetch)

    summary = collect_au_sources.collect_sources(
        target_urls=["https://www.oaic.gov.au/privacy"],
        output_jsonl=str(tmp_path / "oaic_chunks.jsonl"),
        output_summary=str(tmp_path / "oaic_summary.json"),
        state_path=str(tmp_path / "oaic_state.json"),
        rejected_output_path=str(tmp_path / "rejected.jsonl"),
        include_local_pdfs=False,
        incremental=False,
    )

    assert summary["discovery_exit_reason"] == "target_chunk_threshold_reached"
    assert summary["discovered_sub_links_total"] == 10
    assert summary["discovered_chunks_total"] >= 20
    assert summary["chunks_total"] >= 20


def test_collect_sources_uses_sitemap_when_hub_fetch_forbidden(monkeypatch, tmp_path):
    monkeypatch.setattr(collect_au_sources, "COLLECT_REQUIRE_STEALTH", False)
    monkeypatch.setattr(collect_au_sources.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(collect_au_sources.random, "uniform", lambda _a, _b: 2.5)

    def fake_fetch(url, timeout=20):
        raise RuntimeError("403 Forbidden")

    def fake_sitemap(_sitemap_url, limit=10):
        return ["https://www.oaic.gov.au/privacy/guidance-and-advice/test-page"]

    def fake_fetch_followup(url, timeout=20):
        if url.endswith("test-page"):
            return (
                "AFP and ACIC verify applicant identity and conviction history for Australian police check compliance. "
                "This supports lawful screening decisions and result handling across employer workflows. " * 14,
                "html",
                [],
                "<html><body><main><p>test</p></main></body></html>",
            )
        return fake_fetch(url, timeout=timeout)

    monkeypatch.setattr(collect_au_sources, "_fetch_url", fake_fetch_followup)
    monkeypatch.setattr(collect_au_sources, "_discover_via_sitemap", fake_sitemap)

    summary = collect_au_sources.collect_sources(
        target_urls=["https://www.oaic.gov.au/privacy"],
        output_jsonl=str(tmp_path / "oaic_chunks.jsonl"),
        output_summary=str(tmp_path / "oaic_summary.json"),
        state_path=str(tmp_path / "oaic_state.json"),
        rejected_output_path=str(tmp_path / "rejected.jsonl"),
        include_local_pdfs=False,
        incremental=False,
    )

    assert summary["errors_total"] >= 1
    assert summary["discovered_sub_links_total"] == 1
    assert summary["chunks_total"] > 0


def test_collect_sources_reports_manual_pdf_guidance_when_sitemap_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(collect_au_sources, "COLLECT_REQUIRE_STEALTH", False)
    monkeypatch.setattr(collect_au_sources.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(collect_au_sources.random, "uniform", lambda _a, _b: 2.5)

    def fake_fetch(url, timeout=20):
        raise RuntimeError("403 Forbidden")

    def fake_sitemap(_sitemap_url, limit=10):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(collect_au_sources, "_fetch_url", fake_fetch)
    monkeypatch.setattr(collect_au_sources, "_discover_via_sitemap", fake_sitemap)

    summary = collect_au_sources.collect_sources(
        target_urls=["https://www.oaic.gov.au/privacy"],
        output_jsonl=str(tmp_path / "oaic_chunks.jsonl"),
        output_summary=str(tmp_path / "oaic_summary.json"),
        state_path=str(tmp_path / "oaic_state.json"),
        rejected_output_path=str(tmp_path / "rejected.jsonl"),
        include_local_pdfs=False,
        incremental=False,
    )

    assert any(
        item.get("error") == "Vui lòng tải thủ công trang này dưới dạng PDF vào thư mục data/raw/pdfs"
        for item in summary["errors"]
    )
