import math

from dvl.confidence import span_confidence

TOKS = ['{"', 'category', '":"', 'dis', 'aster', '","', 'incident', '_type', '":"',
        'fl', 'ood', '","', 'severity', '":"', 'mild', '"}']


def test_product_of_value_tokens():
    probs = [1.0] * len(TOKS)
    probs[9], probs[10] = 0.9, 0.5  # 'fl', 'ood'
    assert math.isclose(span_confidence(TOKS, probs), 0.45)


def test_ignores_other_fields():
    probs = [0.1] * len(TOKS)
    probs[9] = probs[10] = 1.0
    assert math.isclose(span_confidence(TOKS, probs), 1.0)


def test_token_straddling_quote_counts():
    toks = ['{"incident_type":"', 'flood"', '}']
    assert math.isclose(span_confidence(toks, [1.0, 0.8, 1.0]), 0.8)


def test_missing_key_returns_none():
    assert span_confidence(['ไม่รู้'], [0.9]) is None


def test_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        span_confidence(['a'], [0.1, 0.2])
