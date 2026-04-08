from app.retriever import retrieve_top_k, retrieve_with_guard


def test_retrieve_top_k(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"
    doc1.write_text("police check la gi va vi sao quan trong", encoding="utf-8")
    doc2.write_text("huong dan onboarding employer", encoding="utf-8")

    results = retrieve_top_k("police check", str(tmp_path), top_k=2)
    assert len(results) >= 1
    assert results[0].doc_id == "doc1"


def test_retrieve_prioritizes_processing_time_query(tmp_path):
    time_doc = tmp_path / "doc_time.txt"
    general_doc = tmp_path / "doc_general.txt"
    time_doc.write_text("processing time is usually 1 to 3 days", encoding="utf-8")
    general_doc.write_text("police check helps employer reduce risk", encoding="utf-8")

    results = retrieve_top_k("How long does a police check take?", str(tmp_path), top_k=2)
    assert results
    assert results[0].doc_id == "doc_time"


def test_retrieve_finds_required_documents_query(tmp_path):
    req_doc = tmp_path / "doc_required_documents.txt"
    other_doc = tmp_path / "doc_other.txt"
    req_doc.write_text("required documents include photo id and proof of address", encoding="utf-8")
    other_doc.write_text("candidate experience matters in recruitment", encoding="utf-8")

    results = retrieve_top_k("What documents are required?", str(tmp_path), top_k=2)
    assert results
    assert results[0].doc_id == "doc_required_documents"


def test_retrieve_metadata_reranks_compliance_query(tmp_path):
        compliance_doc = tmp_path / "doc_05_compliance_note.txt"
        documents_doc = tmp_path / "doc_07_required_documents.txt"
        compliance_doc.write_text("compliance checklist for recruitment process", encoding="utf-8")
        documents_doc.write_text("required documents include photo id", encoding="utf-8")

        metadata_path = tmp_path / "metadata.json"
        metadata_path.write_text(
                """
[
    {
        "file_stem": "doc_05_compliance_note",
        "topic": "compliance",
        "document_type": "checklist",
        "approved": true
    },
    {
        "file_stem": "doc_07_required_documents",
        "topic": "required documents",
        "document_type": "requirements",
        "approved": true
    }
]
                """.strip(),
                encoding="utf-8",
        )

        results = retrieve_top_k(
                "Recruitment compliance checklist",
                str(tmp_path),
                top_k=2,
                metadata_path=str(metadata_path),
        )
        assert results
        assert results[0].doc_id == "doc_05_compliance_note"


def test_retrieve_with_guard_out_of_domain(tmp_path):
    doc = tmp_path / "doc_01_police_check.txt"
    doc.write_text("police check for recruitment", encoding="utf-8")

    decision = retrieve_with_guard("Write a travel blog about Vietnam", str(tmp_path), metadata_path=None)
    assert decision.status == "out_of_domain"
    assert decision.docs == []


def test_retrieve_with_guard_low_confidence(tmp_path):
    doc = tmp_path / "doc_01_police_check.txt"
    doc.write_text("police check", encoding="utf-8")

    decision = retrieve_with_guard(
        "police check",
        str(tmp_path),
        metadata_path=None,
        min_top_score=10,
    )
    assert decision.status == "low_confidence"
