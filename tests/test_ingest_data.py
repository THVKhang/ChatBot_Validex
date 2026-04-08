from app.ingest_data import ingest_raw_documents


def test_ingest_raw_documents_creates_processed_and_metadata(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    metadata_file = tmp_path / "metadata" / "documents.json"

    raw_dir.mkdir(parents=True)
    (raw_dir / "doc_test_example.txt").write_text("  sample   content\nwith  spaces ", encoding="utf-8")

    result = ingest_raw_documents(
        raw_dir=str(raw_dir),
        processed_dir=str(processed_dir),
        metadata_path=str(metadata_file),
        source="unit_test",
    )

    assert result["added"] == 1
    assert (processed_dir / "doc_test_example.txt").exists()
    assert metadata_file.exists()
