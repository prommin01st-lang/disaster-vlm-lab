import math

from dvl.metrics import compute_metrics


def g(t, c="disaster", s="mild"):
    return {"incident_type": t, "category": c, "severity": s}


def p(t, c="disaster", s="mild", valid=True, conf=1.0):
    return {"incident_type": t, "category": c, "severity": s, "valid": valid, "confidence": conf}


def test_perfect():
    gold = [g("flood"), g("no_incident", None, "none")]
    m = compute_metrics(gold, [p("flood"), p("no_incident", None, "none")])
    assert m["type_accuracy"] == 1 and m["type_macro_f1"] == 1
    assert m["false_alarm_rate"] == 0 and m["miss_rate"] == 0 and m["json_valid_rate"] == 1


def test_false_alarm_and_miss():
    gold = [g("no_incident", None, "none"), g("no_incident", None, "none"), g("flood"), g("storm")]
    pred = [p("flood"), p("unsure", "other"), p("no_incident", None, "none"), p("storm")]
    m = compute_metrics(gold, pred)
    assert m["false_alarm_rate"] == 0.5   # unsure ไม่นับเป็น false alarm
    assert m["miss_rate"] == 0.5
    assert m["unsure_rate"] == 0.25


def test_macro_f1_over_gold_classes():
    gold = [g("flood"), g("flood"), g("storm")]
    pred = [p("flood"), p("storm"), p("storm")]
    m = compute_metrics(gold, pred)
    # flood P=1 R=.5 F1=.667 ; storm P=.5 R=1 F1=.667
    assert math.isclose(m["type_macro_f1"], 2 / 3, rel_tol=1e-6)
    assert m["per_type"]["flood"]["support"] == 2


def test_json_valid_rate():
    m = compute_metrics([g("flood")] * 2, [p("flood"), p("unsure", "other", valid=False)])
    assert m["json_valid_rate"] == 0.5


def test_ece_perfectly_calibrated_is_zero():
    gold = [g("flood")] * 4
    pred = [p("flood", conf=1.0)] * 4
    assert compute_metrics(gold, pred)["ece"] == 0


def test_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_metrics([g("flood")], [])
