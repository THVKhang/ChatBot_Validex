"""Hybrid search tests against a real PostgreSQL + pgvector database.

The existing suite only ever mocked PGVectorRepository, so it asserted that the
pipeline forwards whatever the mock returns — a tautology. Every real defect
lived below that line and shipped undetected:

  - the RRF fusion used UNION ALL, so a chunk matched by BOTH branches never
    accumulated both reciprocal-rank terms and duplicate rows were returned
  - `fts_content` was missing from initialize_schema(), so the keyword branch
    raised on every query and the whole hybrid search silently degraded
  - websearch_to_tsquery ANDs bare terms, so a multi-word topic matched nothing
    and the keyword branch never contributed at all

These tests skip when DATABASE_URL is unset so CI without a database stays green.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="needs a live PostgreSQL+pgvector database",
)

from app.config import settings  # noqa: E402
from app.vector_repository import (  # noqa: E402
    PGVectorRepository,
    _safe_table,
    build_keyword_tsquery,
)

TOPIC = "spent convictions scheme Crimes Act"


@pytest.fixture(scope="module")
def repo():
    return PGVectorRepository()


@pytest.fixture(scope="module")
def query_vector():
    from app.local_semantics import get_embedding
    return get_embedding(TOPIC).tolist()


@pytest.fixture(scope="module")
def rows(repo, query_vector):
    return repo.hybrid_search(settings.pgvector_table, TOPIC, query_vector, top_k=8)


class TestSchema:
    def test_fts_content_column_exists(self, repo):
        """Without it the keyword branch raises and hybrid search degrades."""
        with repo.db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = %s AND column_name = 'fts_content'",
                    (settings.pgvector_table,),
                )
                assert cur.fetchone() is not None, "fts_content column missing"

    def test_query_vector_matches_table_dimension(self, repo, query_vector):
        """A mismatch here makes every similarity score meaningless."""
        declared = repo.table_embedding_dimension(settings.pgvector_table)
        assert declared is not None
        assert len(query_vector) == declared

    def test_embeddings_come_from_one_provider(self, repo):
        """Vectors from different providers are not comparable to each other."""
        providers = repo.stored_embedding_providers(settings.pgvector_table) - {"unknown"}
        assert len(providers) <= 1, f"mixed embedding providers: {providers}"


class TestKeywordQueryBuilder:
    def test_multi_word_topic_becomes_an_or_query(self):
        """websearch_to_tsquery ANDs bare terms; ORing is what makes it match."""
        assert build_keyword_tsquery(TOPIC) == "spent OR convictions OR scheme OR Crimes OR Act"

    def test_short_tokens_are_dropped(self):
        assert build_keyword_tsquery("a police check in NSW") == "police OR check OR NSW"

    def test_empty_query_falls_back_to_input(self):
        assert build_keyword_tsquery("") == ""


class TestRRFFusion:
    def test_returns_results(self, rows):
        assert rows, "hybrid search returned nothing for an in-corpus topic"

    def test_no_duplicate_chunks(self, rows):
        """UNION ALL used to emit the same chunk once per matching branch."""
        ids = [r["chunk_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_both_branches_actually_fire(self, rows):
        """If only one branch ever matches, this is not a hybrid search."""
        assert any(r["semantic_rank"] for r in rows), "semantic branch never matched"
        assert any(r["keyword_rank"] for r in rows), "keyword branch never matched"

    def test_dual_match_outscores_single_match(self, rows):
        """The whole point of RRF: matching both branches must pay twice."""
        both = [r for r in rows if r["semantic_rank"] and r["keyword_rank"]]
        single = [r for r in rows if not (r["semantic_rank"] and r["keyword_rank"])]
        if not both or not single:
            pytest.skip("need one dual-matched and one single-matched chunk")
        assert min(float(r["rrf_score"]) for r in both) > max(
            float(r["rrf_score"]) for r in single
        )

    def test_rrf_score_matches_the_reciprocal_rank_formula(self, rows):
        for r in rows:
            expected = 0.0
            if r["semantic_rank"]:
                expected += 1.0 / (60 + int(r["semantic_rank"]))
            if r["keyword_rank"]:
                expected += 1.0 / (60 + int(r["keyword_rank"]))
            assert float(r["rrf_score"]) == pytest.approx(expected, rel=1e-6)

    def test_results_are_ordered_by_rrf_descending(self, rows):
        scores = [float(r["rrf_score"]) for r in rows]
        assert scores == sorted(scores, reverse=True)


class TestFilters:
    def test_repealed_legislation_is_excluded(self, rows):
        """Citing repealed law in compliance content is the worst failure mode."""
        assert all(r["status"] == "in_force" for r in rows)

    def test_only_approved_chunks_are_returned(self, rows):
        assert all(r["approved"] for r in rows)

    def test_similarity_floor_is_applied(self, repo, query_vector):
        strict = repo.hybrid_search(
            settings.pgvector_table, TOPIC, query_vector, top_k=8, min_similarity=0.99
        )
        # Only keyword-only hits (no cosine) may survive an impossible floor.
        assert all(r["semantic_rank"] is None for r in strict)

    def test_filter_values_are_bound_not_interpolated(self, repo, query_vector):
        """A quote in a filter value must not be able to break the statement."""
        out = repo.hybrid_search(
            settings.pgvector_table, TOPIC, query_vector, top_k=3,
            status_filter="in_force' OR '1'='1",
        )
        assert out == []


class TestTableNameGuard:
    def test_accepts_the_configured_table(self):
        assert _safe_table(settings.pgvector_table) == settings.pgvector_table

    @pytest.mark.parametrize("bad", [
        "validex_knowledge; DROP TABLE users",
        "public.validex_knowledge",
        "",
        "1_starts_with_digit",
    ])
    def test_rejects_unsafe_identifiers(self, bad):
        with pytest.raises(ValueError):
            _safe_table(bad)


class TestParentRetrieval:
    def test_parent_content_resolves_when_present(self, repo, rows):
        with_parent = [r for r in rows if r.get("parent_id")]
        if not with_parent:
            pytest.skip("no parent-linked chunks in the corpus")
        content = repo.get_parent_content(settings.pgvector_table, with_parent[0]["parent_id"])
        assert content and len(content) > 0
