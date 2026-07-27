"""Tests for the Regulatory Fact-Checker (golden_facts.py).

Covers:
- Correct numbers → no violations
- Wrong numbers → violations detected with correct values
- Auto-correction of wrong numbers
- Edge cases: no numbers, empty draft, multiple facts
- Integration with Editor Node Layer 0.5
"""

import pytest

from app.golden_facts import (
    GOLDEN_FACTS,
    GoldenFact,
    FactViolation,
    check_facts,
    auto_correct_facts,
)


class TestCheckFacts:
    """Test the check_facts() scanner."""

    def test_empty_draft_returns_no_violations(self):
        assert check_facts("") == []

    def test_none_draft_returns_no_violations(self):
        assert check_facts(None) == []

    def test_no_matching_patterns_returns_no_violations(self):
        draft = "This is a blog about cooking recipes. Nothing about identity checks."
        assert check_facts(draft) == []

    # ── 100-Point Identity Check Facts ─────────────────────────

    def test_foreign_passport_correct_70_no_violation(self):
        draft = "A foreign passport is worth 70 points in the identity check."
        assert check_facts(draft) == []

    def test_foreign_passport_wrong_30_detected(self):
        draft = "A foreign passport gives you 30 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-001"
        assert violations[0].found_value == "30"
        assert violations[0].correct_value == "70"

    def test_foreign_passport_wrong_60_detected(self):
        draft = "An overseas passport is worth 60 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-001"
        assert violations[0].found_value == "60"

    def test_international_passport_wrong_detected(self):
        draft = "Your international passport gives 50 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-001"

    def test_australian_bank_statement_correct_25_no_violation(self):
        draft = "An Australian bank statement is worth 25 points."
        assert check_facts(draft) == []

    def test_australian_bank_statement_wrong_30_detected(self):
        draft = "An Australian bank statement provides 30 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-002"
        assert violations[0].found_value == "30"
        assert violations[0].correct_value == "25"

    def test_australian_passport_correct_70_no_violation(self):
        draft = "An Australian passport is worth 70 points."
        assert check_facts(draft) == []

    def test_australian_passport_wrong_50_detected(self):
        draft = "An Australian passport gives you 50 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-003"

    def test_drivers_licence_correct_40_no_violation(self):
        draft = "A driver's licence is worth 40 points."
        assert check_facts(draft) == []

    def test_drivers_licence_wrong_25_detected(self):
        draft = "Your driver's license is worth 25 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-005"
        assert violations[0].found_value == "25"
        assert violations[0].correct_value == "40"

    def test_medicare_card_correct_25_no_violation(self):
        draft = "A Medicare card is worth 25 points."
        assert check_facts(draft) == []

    def test_medicare_card_wrong_20_detected(self):
        draft = "A Medicare card provides 20 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-006"

    def test_birth_certificate_correct_70_no_violation(self):
        draft = "An Australian birth certificate is worth 70 points."
        assert check_facts(draft) == []

    def test_birth_certificate_wrong_50_detected(self):
        draft = "An Australian birth certificate provides 50 points."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-004"

    # ── Police Check Facts ─────────────────────────────────────

    def test_wwcc_validity_correct_5_no_violation(self):
        draft = "A Working With Children Check is valid for 5 years."
        assert check_facts(draft) == []

    def test_wwcc_validity_wrong_3_detected(self):
        draft = "A WWCC lasts for 3 years before renewal."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "PC-001"
        assert violations[0].found_value == "3"
        assert violations[0].correct_value == "5"

    # ── Multiple Violations ────────────────────────────────────

    def test_multiple_wrong_facts_all_detected(self):
        draft = (
            "For the 100-point check: A foreign passport gives 30 points, "
            "and an Australian bank statement provides 30 points."
        )
        violations = check_facts(draft)
        fact_ids = {v.fact_id for v in violations}
        assert "ID-001" in fact_ids  # foreign passport
        assert "ID-002" in fact_ids  # bank statement

    def test_mixed_correct_and_wrong_only_wrong_detected(self):
        draft = (
            "A foreign passport is worth 70 points (correct). "
            "An Australian bank statement gives you 30 points (wrong)."
        )
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-002"

    # ── Case Insensitivity ─────────────────────────────────────

    def test_case_insensitive_matching(self):
        draft = "A FOREIGN PASSPORT gives 30 POINTS."
        violations = check_facts(draft)
        assert len(violations) == 1
        assert violations[0].fact_id == "ID-001"


class TestAutoCorrectFacts:
    """Test the auto_correct_facts() function."""

    def test_no_violations_returns_unchanged_draft(self):
        draft = "A foreign passport is worth 70 points."
        corrected, fixed = auto_correct_facts(draft)
        assert corrected == draft
        assert fixed == []

    def test_wrong_number_auto_corrected(self):
        draft = "A foreign passport gives 30 points in the check."
        corrected, fixed = auto_correct_facts(draft)
        assert "70 points" in corrected
        assert "30 points" not in corrected
        assert len(fixed) == 1
        assert fixed[0].fact_id == "ID-001"

    def test_bank_statement_30_corrected_to_25(self):
        draft = "An Australian bank statement provides 30 points."
        corrected, fixed = auto_correct_facts(draft)
        assert "25 points" in corrected
        assert "30 points" not in corrected

    def test_multiple_corrections(self):
        draft = (
            "A foreign passport gives 60 points. "
            "An Australian bank statement provides 30 points."
        )
        corrected, fixed = auto_correct_facts(draft)
        assert "70 points" in corrected
        assert "25 points" in corrected
        assert len(fixed) == 2

    def test_empty_draft_returns_empty(self):
        corrected, fixed = auto_correct_facts("")
        assert corrected == ""
        assert fixed == []


class TestGoldenFactsRegistry:
    """Test the GOLDEN_FACTS registry itself."""

    def test_registry_is_populated(self):
        assert len(GOLDEN_FACTS) >= 10

    def test_all_facts_have_required_fields(self):
        for fact in GOLDEN_FACTS:
            assert fact.id, f"Fact missing id"
            assert fact.category, f"Fact {fact.id} missing category"
            assert fact.pattern, f"Fact {fact.id} missing pattern"
            assert fact.source, f"Fact {fact.id} missing source"
            assert fact.correction, f"Fact {fact.id} missing correction"

    def test_all_fact_ids_unique(self):
        ids = [f.id for f in GOLDEN_FACTS]
        assert len(ids) == len(set(ids)), f"Duplicate fact IDs found"

    def test_all_patterns_compile(self):
        import re
        for fact in GOLDEN_FACTS:
            try:
                re.compile(fact.pattern, re.IGNORECASE)
            except re.error as exc:
                pytest.fail(f"Fact {fact.id} has invalid regex: {exc}")

    def test_all_patterns_have_capture_group(self):
        import re
        for fact in GOLDEN_FACTS:
            compiled = re.compile(fact.pattern, re.IGNORECASE)
            assert compiled.groups >= 1, (
                f"Fact {fact.id} pattern has no capture group"
            )


class TestRealWorldScenarios:
    """Test with realistic LLM-generated draft snippets."""

    def test_classic_hallucination_bank_statement_30_points(self):
        """The exact scenario that prompted this feature."""
        draft = (
            "## 100-Point Identity Check\n\n"
            "As an international student, you have:\n"
            "- Foreign Passport: 70 points\n"
            "- Australian bank statement: 30 points\n\n"
            "Total: 100 points. You meet the requirement!"
        )
        violations = check_facts(draft)
        assert any(v.fact_id == "ID-002" for v in violations)

        corrected, _ = auto_correct_facts(draft)
        assert "25 points" in corrected
        assert "30 points" not in corrected

    def test_correct_draft_passes_cleanly(self):
        """A factually correct draft should produce no violations."""
        draft = (
            "## 100-Point Identity Check\n\n"
            "Primary documents (70 points each):\n"
            "- Australian passport: 70 points\n"
            "- Foreign passport: 70 points\n\n"
            "Secondary documents (25-40 points):\n"
            "- Driver's licence: 40 points\n"
            "- Medicare card: 25 points\n"
            "- Australian bank statement: 25 points\n"
        )
        assert check_facts(draft) == []

    def test_draft_without_numbers_is_safe(self):
        """A draft that mentions documents but no point values should pass."""
        draft = (
            "You can use your foreign passport and bank statement "
            "for the identity check. These are commonly accepted documents."
        )
        assert check_facts(draft) == []
