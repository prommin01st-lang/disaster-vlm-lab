import json

import pytest

from dvl.schema import Prediction, parse_output, target_json, to_api


def test_target_json_exact_format():
    assert target_json("flood", "mild") == '{"category":"disaster","incident_type":"flood","severity":"mild"}'


def test_target_json_no_incident():
    assert target_json("no_incident", "none") == '{"category":null,"incident_type":"no_incident","severity":"none"}'


def test_target_json_rejects_bad_combo():
    with pytest.raises(ValueError):
        target_json("flood", "none")
    with pytest.raises(ValueError):
        target_json("no_incident", "severe")
    with pytest.raises(KeyError):
        target_json("earthquake", "mild")


def test_parse_roundtrip():
    p = parse_output(target_json("forest_fire", "severe"))
    assert p == Prediction("fire_hazard", "forest_fire", "severe", True, None)


@pytest.mark.parametrize("text", [
    '```json\n{"category":"disaster","incident_type":"flood","severity":"mild"}\n```',
    'คำตอบ: {"category":"disaster","incident_type":"flood","severity":"mild"} เพราะน้ำสูง',
    '{"category": "disaster", "incident_type": "Flood", "severity": "MILD"}',
])
def test_parse_tolerates_noise_and_case(text):
    p = parse_output(text)
    assert p.valid and p.incident_type == "flood" and p.severity == "mild"


def test_parse_fixes_inconsistent_category():
    p = parse_output('{"category":"accident","incident_type":"flood","severity":"mild"}')
    assert p.valid and p.category == "disaster"


def test_parse_forces_none_severity_for_no_incident():
    p = parse_output('{"category":null,"incident_type":"no_incident","severity":"mild"}')
    assert p.valid and p.severity == "none"


def test_parse_incident_with_none_severity_becomes_mild():
    p = parse_output('{"category":"disaster","incident_type":"flood","severity":"none"}')
    assert p.valid and p.severity == "mild"


@pytest.mark.parametrize("text,err", [
    ('', "no_json"),
    ('ไม่ทราบ', "no_json"),
    ('{"category":"disaster","incident_type":"flood"', "bad_json"),
    ('{"category":"disaster","incident_type":"earthquake","severity":"mild"}', "unknown_type"),
    ('{"category":"disaster","incident_type":"น้ำท่วม","severity":"mild"}', "unknown_type"),
    ('{"category":"disaster","incident_type":"flood","severity":"extreme"}', "unknown_severity"),
    ('["flood"]', "no_json"),
])
def test_parse_invalid_falls_back_to_unsure(text, err):
    p = parse_output(text)
    assert p == Prediction("other", "unsure", "mild", False, err)


def test_to_api():
    p = parse_output(target_json("flood", "severe"))
    assert to_api(p, 0.912345) == {"category": "disaster", "incident_type": "flood",
                                   "severity": "severe", "confidence": 0.912}
    json.dumps(to_api(p, 0.5))  # serialize ได้
