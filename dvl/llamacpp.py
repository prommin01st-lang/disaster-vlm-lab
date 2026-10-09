"""ไคลเอนต์ llama-server (OpenAI-compatible) ที่ใช้ร่วมกัน: scripts/predict_gguf.py, scripts/abstain_sweep.py, api/app.py
ข้อความมาจาก dvl.prompt.build_messages เสมอ · กติกา abstain: no_incident ที่ P(เป็นเหตุ) >= T → unsure"""
import base64, io, json, math, mimetypes, re, urllib.request
from pathlib import Path

from PIL import Image

from dvl.confidence import span_confidence


def png_data_url(img: Image.Image) -> str:
    """lossless — pixel ที่ส่งตรงกับที่ถอดได้ (encode JPEG ซ้ำอีกรอบทำให้คำตอบเปลี่ยน ~5% บน gold)"""
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _image_url(image) -> str:
    """image = path ของไฟล์ (ส่ง bytes ตามไฟล์) หรือ data URL สำเร็จรูป (ใช้ตรง ๆ)"""
    if isinstance(image, str) and image.startswith("data:"):
        return image
    p = Path(image)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def to_openai(messages: list[dict]) -> list[dict]:
    """build_messages (รูป HF: {"type":"image","image":<path|data URL>}) → รูป OpenAI (image_url เป็น data URL)"""
    out = []
    for m in messages:
        parts = []
        for c in m["content"]:
            if c["type"] == "text":
                parts.append({"type": "text", "text": c["text"]})
            else:
                parts.append({"type": "image_url", "image_url": {"url": _image_url(c["image"])}})
        out.append({"role": m["role"], "content": parts})
    return out


def request_body(messages: list[dict]) -> dict:
    return {"messages": to_openai(messages), "temperature": 0, "max_tokens": 64, "logprobs": True, "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False}}


def value_alternatives(lp: list[dict], key: str) -> list[list]:
    """top_logprobs ของ token แรกของค่า <key> → [[token, logprob], ...]
    category มาก่อน incident_type และเป็น null เมื่อไม่ใช่เหตุ → จุดตัดสิน "เหตุ/ไม่ใช่เหตุ" อยู่ที่ token นี้"""
    m = re.search(rf'"{key}"\s*:\s*"?', "".join(t["token"] for t in lp))
    pos = 0
    for t in lp if m else []:
        if pos >= m.end():
            return [[a["token"], a["logprob"]] for a in t.get("top_logprobs") or []]
        pos += len(t["token"])
    return []


def parse_response(resp: dict) -> tuple[str, float | None, bool]:
    """→ (text, confidence, has_thinking) — has_thinking = มี <think>/reasoning_content ที่ไม่ว่าง"""
    ch = resp["choices"][0]
    text = ch["message"].get("content") or ""
    reasoning = ch["message"].get("reasoning_content") or ""
    lp = (ch.get("logprobs") or {}).get("content") or []
    conf = None
    if lp:
        conf = span_confidence([t["token"] for t in lp], [math.exp(t["logprob"]) for t in lp])
    return text, conf, bool(reasoning.strip()) or "<think>" in text


def response_logprobs(resp: dict) -> list[dict]:
    return (resp["choices"][0].get("logprobs") or {}).get("content") or []


def post(url: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def p_incident(p: dict) -> float:
    """P(เป็นเหตุ) = 1 − P(category = null) จาก category_alts (top-5) — null หลุด top-5 → 1.0 (ส่งให้คนดู)"""
    p_null = sum(math.exp(lp) for tok, lp in p["category_alts"] if tok.strip() == "null")
    return max(0.0, 1.0 - p_null)


def gate_fires(p: dict, t: float | None) -> bool:
    return t is not None and p["incident_type"] == "no_incident" and p_incident(p) >= t


def abstain(p: dict, t: float | None) -> dict:
    if gate_fires(p, t):
        return {**p, "category": "other", "incident_type": "unsure", "severity": "mild"}
    return p
