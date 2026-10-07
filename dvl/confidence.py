"""confidence = ความน่าจะเป็นที่โมเดลให้กับ token ของค่า incident_type (ไม่ใช่เลขที่โมเดลเขียนเอง)"""
import math
import re


def span_confidence(tokens: list[str], probs: list[float], key: str = "incident_type") -> float | None:
    if len(tokens) != len(probs):
        raise ValueError("tokens/probs length mismatch")
    text = "".join(tokens)
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)
    if not m:
        return None
    lo, hi = m.span(1)
    pos, picked = 0, []
    for tok, p in zip(tokens, probs):
        start, end = pos, pos + len(tok)
        if start < hi and end > lo:  # token คาบเกี่ยวช่วงค่า
            picked.append(p)
        pos = end
    return math.prod(picked) if picked else None
