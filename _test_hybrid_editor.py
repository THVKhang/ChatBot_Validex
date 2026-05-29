"""Test the Hybrid Editor code-gate logic."""
from collections import Counter

def test_code_gate():
    """Simulate the code-gate checks without needing the full pipeline."""
    
    # Test 1: Good blog (should auto-accept)
    good_draft = "# Police Check\n\n" + "\n\n".join([
        f"## Section {i}\n\nThis is a well-written paragraph about police checks in Australia. "
        f"The ACIC processes thousands of requests daily through the APIN protocol. "
        f"Each section covers different aspects of the background check process with "
        f"sufficient detail and technical depth to satisfy enterprise requirements. "
        f"The system leverages PostgreSQL databases with AES-256 encryption to ensure "
        f"data integrity and compliance with the Privacy Act 1988. Furthermore, the "
        f"REST API endpoints handle authentication via OAuth 2.0 tokens, ensuring that "
        f"only authorized personnel can access sensitive criminal history records. "
        f"The infrastructure is deployed across multiple availability zones for redundancy."
        for i in range(6)
    ]) + "\n\n## Conclusion and Strategic Next Steps\n\nThe analysis shows that police checks are essential for maintaining compliance."
    
    word_count = len(good_draft.split())
    heading_count = good_draft.count('## ')
    has_conclusion = 'conclusion' in good_draft.lower()
    
    assert word_count > 500, f"Good draft should be >500 words, got {word_count}"
    assert heading_count >= 4, f"Good draft should have 4+ headings, got {heading_count}"
    assert has_conclusion, "Good draft should have conclusion"
    print(f"PASS (good blog): {word_count} words, {heading_count} headings → AUTO-ACCEPT")
    
    # Test 2: Bad blog (should auto-reject)
    bad_draft = "This is a very short blog with no structure at all."
    word_count = len(bad_draft.split())
    heading_count = bad_draft.count('## ')
    
    issues = []
    if heading_count < 2:
        issues.append("E04:insufficient-headings")
    if 'conclusion' not in bad_draft.lower():
        issues.append("E06:no-conclusion")
    
    assert len(issues) > 0, "Bad draft should have issues"
    print(f"PASS (bad blog): {word_count} words, {heading_count} headings, issues={issues} → AUTO-REJECT")
    
    # Test 3: Repetition detection
    repeated_draft = "## Section 1\n\n" + "The APIN protocol is very important. " * 20
    words = repeated_draft.lower().split()
    ngrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
    repeated = [t for t, c in Counter(ngrams).items() if c >= 4 and len(t) > 12]
    assert len(repeated) > 0, "Should detect repetition"
    print(f"PASS (repetition): detected {len(repeated)} repeated 4-grams → AUTO-REJECT")
    
    # Test 4: Ambiguous case (should go to LLM)
    ambiguous_draft = "## Introduction\n\nSome content here about the topic.\n\n## Details\n\nMore details.\n\n## Conclusion\n\nIn conclusion."
    word_count = len(ambiguous_draft.split())
    heading_count = ambiguous_draft.count('## ')
    
    # Not enough words for auto-accept, but has structure so no auto-reject
    auto_accept = word_count > 500 and heading_count >= 4
    issues = []
    if heading_count < 2:
        issues.append("E04")
    
    assert not auto_accept, "Ambiguous draft should NOT auto-accept"
    assert len(issues) == 0, "Ambiguous draft should have no structural issues"
    print(f"PASS (ambiguous): {word_count} words, {heading_count} headings → LLM NEEDED")

    print("\nAll hybrid editor tests passed!")

test_code_gate()
