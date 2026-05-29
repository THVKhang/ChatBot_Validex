import re

# Lookup table for common AI-generated gerunds
known = {
    "securing": "secures",
    "facilitating": "facilitates",
    "managing": "manages",
    "handling": "handles",
    "monitoring": "monitors",
    "processing": "processes",
    "ensuring": "ensures",
    "protecting": "protects",
}

def _gerund_to_verb(m):
    gerund = m.group(1).lower()
    if gerund in known:
        return known[gerund]
    return f"is responsible for {gerund}"

test_cases = [
    ('The PKI infrastructure plays a crucial role in securing the data', 'secures'),
    ('The API plays a crucial role in facilitating the exchange', 'facilitates'),
    ('The system plays a crucial role in managing resources', 'manages'),
    ('The protocol plays a vital role in handling traffic', 'handles'),
    ('The framework plays a critical role in the ecosystem', 'is responsible for'),
]

for draft, expected_word in test_cases:
    for fluff in ['plays a crucial role in', 'plays a critical role in', 'plays a vital role in']:
        pattern = re.escape(fluff) + r'\s+(\w+ing)\b'
        draft = re.sub(pattern, _gerund_to_verb, draft, flags=re.IGNORECASE)
        draft = draft.replace(fluff, 'is responsible for')
    
    assert expected_word in draft, f'FAIL: "{expected_word}" not found in "{draft}"'
    print(f'PASS: "{draft}"')

# Test REDACTED masking
for tag_case in [
    'The [REDACTED_HR_TERM]-based service level agreement',
    'The [REDACTED_HR_TERM] agreements are in place',
    'Multiple [REDACTED_HR_TERM]s exist',
]:
    result = re.sub(r'\[REDACTED_HR_TERM\]-based', 'operational', tag_case)
    result = re.sub(r'\[REDACTED_HR_TERM\]s?', 'operational', result)
    result = result.replace('[REDACTED_HR_TERM]', 'operational')
    assert '[REDACTED' not in result, f'FAIL: tag leaked in "{result}"'
    print(f'PASS (masking): "{result}"')

print("\nAll tests passed!")
