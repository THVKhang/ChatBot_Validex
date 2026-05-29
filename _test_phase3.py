"""Test Phase 3: Python replacement functions for LLM calls."""
import sys
sys.path.insert(0, '.')

def test_all():
    # Test 1: Query expansion (Python)
    from app.agents.researcher_node import _expand_query
    q1 = _expand_query('police check expiry')
    assert len(q1) >= 3, f'Expected 3+ queries, got {len(q1)}'
    assert 'police check expiry' in q1
    print(f'PASS query_expand (police check expiry): {q1}')

    q2 = _expand_query('wwcc requirements')
    assert len(q2) >= 3
    print(f'PASS query_expand (wwcc): {q2}')

    q3 = _expand_query('visa background check')
    assert len(q3) >= 3
    print(f'PASS query_expand (visa): {q3}')

    # Test 2: Extractive summarize
    from app.agents.researcher_node import _extractive_summarize
    text = (
        'The ACIC processes police checks nationally. '
        'Each check verifies criminal history records. '
        'The system uses APIN protocol for secure data exchange. '
        'Weather in Sydney is sunny today. '
        'Police check results are typically available within 2 business days.'
    )
    summary = _extractive_summarize(text, 'police check', max_output=200)
    assert 'police' in summary.lower() or 'criminal' in summary.lower()
    print(f'PASS extractive_summarize: "{summary[:80]}..."')

    # Test 3: Discovery keyword evaluation
    from app.agents.discovery_agent import _semantic_evaluate_relevance
    r1 = _semantic_evaluate_relevance(
        'https://acic.gov.au/police-check',
        'National Police Check',
        'Apply for background screening and criminal history verification'
    )
    assert r1['score'] >= 5, f'Expected score >= 5, got {r1["score"]}'
    print(f'PASS discovery eval (gov.au): score={r1["score"]}, reason={r1["reason"]}')

    r2 = _semantic_evaluate_relevance(
        'https://random-blog.com/cooking',
        'Best pasta recipes',
        'How to make carbonara at home'
    )
    assert r2['score'] <= 3, f'Expected low score, got {r2["score"]}'
    print(f'PASS discovery eval (irrelevant): score={r2["score"]}, reason={r2["reason"]}')

    # Test 4: Chunk evaluation (collect_au_sources)
    from app.collect_au_sources import _ai_evaluate_chunk
    r3 = _ai_evaluate_chunk(
        'The AFP conducts national police checks for applicants seeking employment clearance. '
        'Criminal history screening is required under Australian legislation.'
    )
    assert r3['score'] >= 5
    print(f'PASS chunk_eval (relevant): score={r3["score"]}, reason={r3["reason"]}')

    r4 = _ai_evaluate_chunk('The weather forecast shows sunny skies across Melbourne this weekend.')
    assert r4['score'] <= 4
    print(f'PASS chunk_eval (irrelevant): score={r4["score"]}, reason={r4["reason"]}')

    print('\nAll Phase 4 unit tests passed!')

test_all()
