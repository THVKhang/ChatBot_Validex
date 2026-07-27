"""Regulatory Fact-Checker — Code-Based Anti-Hallucination for Legal Numbers.

Maintains a registry of verified Australian regulatory facts and scans
generated drafts for incorrect numerical claims.  Pure Python, 0 LLM tokens.

Usage::

    from app.golden_facts import check_facts

    violations = check_facts(draft_text)
    for v in violations:
        print(f"{v.fact_id}: draft says {v.found_value}, correct is {v.correct_value}")
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────────────

@dataclass
class GoldenFact:
    """A single verified regulatory fact."""
    id: str
    category: str
    description: str
    pattern: str                # regex with a capture group for the number
    correct_value: Any          # int, float, or str
    tolerance: int | None       # allowed deviation (0 = exact match, None = skip numeric check)
    source: str                 # official source
    correction: str             # human-readable correction text


@dataclass
class FactViolation:
    """A detected factual error in a draft."""
    fact_id: str
    found_value: str
    correct_value: Any
    source: str
    correction: str
    matched_text: str           # the text span that matched


# ── Golden Facts Registry ────────────────────────────────────────────
# Each entry is a verified fact from Australian government sources.
# Patterns use re.IGNORECASE and expect a capture group (\d+) for the number.

GOLDEN_FACTS: list[GoldenFact] = [
    # ─── 100-Point Identity Check — Document Point Values ───────────
    GoldenFact(
        id="ID-001",
        category="100-point-check",
        description="Foreign passport point value",
        pattern=r"(?:foreign|overseas|international)\s+passport\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=70,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="A foreign/overseas passport is a Primary Document worth 70 points",
    ),
    GoldenFact(
        id="ID-002",
        category="100-point-check",
        description="Australian bank statement point value",
        pattern=r"(?:australian|aus\.?)\s+(?:bank|financial)\s+(?:statement|account)\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=25,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="An Australian bank statement is a Secondary Document worth 25 points",
    ),
    GoldenFact(
        id="ID-003",
        category="100-point-check",
        description="Australian passport point value",
        pattern=r"(?:australian|aus\.?)\s+passport\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=70,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="An Australian passport is a Primary Document worth 70 points",
    ),
    GoldenFact(
        id="ID-004",
        category="100-point-check",
        description="Australian birth certificate point value",
        pattern=r"(?:australian|aus\.?)\s+(?:birth\s+certificate|citizenship\s+certificate)\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=70,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="An Australian birth certificate or citizenship certificate is a Primary Document worth 70 points",
    ),
    GoldenFact(
        id="ID-005",
        category="100-point-check",
        description="Driver's licence point value",
        pattern=r"(?:driver'?s?\s+licen[sc]e|driving\s+licen[sc]e)\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=40,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="An Australian driver's licence is a Secondary Document worth 40 points",
    ),
    GoldenFact(
        id="ID-006",
        category="100-point-check",
        description="Medicare card point value",
        pattern=r"medicare\s+card\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=25,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="A Medicare card is a Secondary Document worth 25 points",
    ),
    GoldenFact(
        id="ID-007",
        category="100-point-check",
        description="Total points required for identity check",
        pattern=r"(?:need|require|must\s+have|total\s+of)\s+(\d+)\s*points?\s+(?:of\s+)?(?:id|identification|identity)",
        correct_value=100,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="The 100-point identity check requires exactly 100 points of identification",
    ),
    GoldenFact(
        id="ID-008",
        category="100-point-check",
        description="Utility bill / rates notice point value",
        pattern=r"(?:utility\s+bill|rates?\s+notice|electricity\s+bill|gas\s+bill|water\s+bill)\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=25,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="A utility bill or rates notice is a Secondary Document worth 25 points",
    ),
    GoldenFact(
        id="ID-009",
        category="100-point-check",
        description="Student ID card point value",
        pattern=r"(?:student\s+(?:id|identification)\s+card)\b[^.]{0,60}?(\d+)\s*points?",
        correct_value=25,
        tolerance=0,
        source="ACIC 100-Point Identity Check Schedule",
        correction="A student ID card is a Secondary Document worth 25 points",
    ),

    # ─── Police Check ──────────────────────────────────────────────
    GoldenFact(
        id="PC-001",
        category="police-check",
        description="WWCC validity period",
        pattern=r"(?:working\s+with\s+children|wwcc)\b[^.]{0,80}?(?:valid|lasts?|expires?)\b[^.]{0,40}?(\d+)\s*years?",
        correct_value=5,
        tolerance=0,
        source="State WWCC legislation (NSW, VIC, QLD, etc.)",
        correction="A Working With Children Check is valid for 5 years in most Australian states",
    ),
    GoldenFact(
        id="PC-002",
        category="police-check",
        description="Spent convictions waiting period (NSW, 10 years)",
        pattern=r"spent\s+conviction[s]?\b[^.]{0,80}?(\d+)\s*years?\s*(?:waiting|period|before)",
        correct_value=10,
        tolerance=0,
        source="Criminal Records Act 1991 (NSW)",
        correction="In NSW, a conviction becomes spent after a 10-year crime-free period (for adults)",
    ),
]


# ── Fact-Checking Engine ─────────────────────────────────────────────

def check_facts(draft: str) -> list[FactViolation]:
    """Scan a draft for numerical claims that contradict verified golden facts.

    Returns a list of FactViolation objects for each incorrect claim found.
    Empty list means no violations detected.
    """
    if not draft:
        return []

    violations: list[FactViolation] = []

    for fact in GOLDEN_FACTS:
        try:
            matches = list(re.finditer(fact.pattern, draft, re.IGNORECASE))
        except re.error as exc:
            logger.error("Regex error in golden fact %s: %s", fact.id, exc)
            continue

        for match in matches:
            # Extract the captured number
            try:
                found_str = match.group(1)
                found_num = int(found_str)
            except (IndexError, ValueError):
                continue

            # Skip numeric check if tolerance is None (text-based facts)
            if fact.tolerance is None:
                continue

            # Check if the number matches the correct value
            correct_num = int(fact.correct_value)
            if abs(found_num - correct_num) > fact.tolerance:
                violation = FactViolation(
                    fact_id=fact.id,
                    found_value=found_str,
                    correct_value=str(fact.correct_value),
                    source=fact.source,
                    correction=fact.correction,
                    matched_text=match.group(0)[:120],
                )
                violations.append(violation)
                logger.warning(
                    "FACT VIOLATION %s: Draft says '%s' = %s points, correct is %s. Source: %s",
                    fact.id, match.group(0)[:60], found_str, fact.correct_value, fact.source,
                )

    return violations


def auto_correct_facts(draft: str) -> tuple[str, list[FactViolation]]:
    """Scan draft and auto-correct wrong numbers. Returns (corrected_draft, violations_fixed).

    Only corrects simple numeric substitutions where the fix is unambiguous.
    """
    violations = check_facts(draft)
    if not violations:
        return draft, []

    corrected = draft
    fixed: list[FactViolation] = []

    for v in violations:
        # Replace the wrong number with the correct one in the matched span
        old_span = v.matched_text
        new_span = old_span.replace(v.found_value, str(v.correct_value), 1)
        if old_span in corrected:
            corrected = corrected.replace(old_span, new_span, 1)
            fixed.append(v)
            logger.info(
                "FACT AUTO-CORRECTED %s: '%s' → '%s'",
                v.fact_id, old_span[:60], new_span[:60],
            )

    return corrected, fixed
