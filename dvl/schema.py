"""JSON เป้าหมายของโมเดล + parser ที่ทนคำตอบเพี้ยน (ไม่ raise)"""
import json
from dataclasses import dataclass

from dvl.catalog import ALLOWED_TYPES, SEVERITIES, category_of


@dataclass(frozen=True)
class Prediction:
    category: str | None
    incident_type: str
    severity: str
    valid: bool
    error: str | None


def _check_combo(incident_type: str, severity: str) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    if (incident_type == "no_incident") != (severity == "none"):
        raise ValueError(f"bad combo {incident_type}/{severity}")


def target_json(incident_type: str, severity: str) -> str:
    category = category_of(incident_type)  # KeyError ถ้าไม่รู้จัก
    _check_combo(incident_type, severity)
    return json.dumps(
        {"category": category, "incident_type": incident_type, "severity": severity},
        ensure_ascii=False, separators=(",", ":"),
    )


def _fallback(error: str) -> Prediction:
    return Prediction("other", "unsure", "mild", False, error)


def parse_output(text: str) -> Prediction:
    start = text.find("{")
    if start < 0:
        return _fallback("no_json")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return _fallback("bad_json")
    if not isinstance(obj, dict):
        return _fallback("no_json")
    t = str(obj.get("incident_type", "")).strip().lower()
    s = str(obj.get("severity", "")).strip().lower()
    if t not in ALLOWED_TYPES:
        return _fallback("unknown_type")
    if s not in SEVERITIES:
        return _fallback("unknown_severity")
    # ซ่อม severity ให้สอดคล้องกับ type (เชื่อ incident_type มากกว่า)
    if t == "no_incident":
        s = "none"
    elif s == "none":
        s = "mild"
    return Prediction(category_of(t), t, s, True, None)


def to_api(pred: Prediction, confidence: float) -> dict:
    return {"category": pred.category, "incident_type": pred.incident_type,
            "severity": pred.severity, "confidence": round(float(confidence), 3)}
