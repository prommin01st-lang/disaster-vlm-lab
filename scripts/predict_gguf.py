"""ทำนาย test_gold ด้วย GGUF ผ่าน llama-server (OpenAI-compatible) → predictions-<TAG>.jsonl (รูปแบบเดียวกับ predict.py)
ใช้: python scripts/predict_gguf.py --tag gguf_q8 [--url http://127.0.0.1:8080] [--split gold|test] [--limit N]
ต้องเปิด llama-server ก่อน (ดู README "Run locally (GGUF)") · ข้อความ = dvl.prompt.build_messages แปลงเป็นรูป OpenAI
confidence = span_confidence จาก logprobs ของ server (softmax ของ logits ก่อน sampling) เหมือน dvl/confidence.py"""
import argparse, base64, json, math, mimetypes, re, subprocess, sys, threading, time, urllib.request
from pathlib import Path

from dvl.confidence import span_confidence
from dvl.prompt import build_messages
from dvl.schema import parse_output, to_api

D = Path("data/dataset_v1")


def to_openai(messages: list[dict]) -> list[dict]:
    """build_messages (รูป HF: {"type":"image","image":<path>}) → รูป OpenAI (image_url เป็น data URL)"""
    out = []
    for m in messages:
        parts = []
        for c in m["content"]:
            if c["type"] == "text":
                parts.append({"type": "text", "text": c["text"]})
            else:
                p = Path(c["image"])
                mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
                url = f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
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


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


class VramPeak(threading.Thread):
    """อ่าน nvidia-smi memory.used ทุก 0.5 s (ทั้งการ์ด รวม desktop) — เก็บค่าสูงสุด"""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                v = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.split()[0])
                self.peak = max(self.peak, v)
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self.stop.wait(0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--split", default="gold", choices=["gold", "test", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="runs/gguf")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(D / {"gold": "test_gold.jsonl", "test": "test.jsonl", "val": "val.jsonl"}[a.split],
                                        encoding="utf-8")]
    rows = rows[:a.limit] if a.limit else rows
    Path(a.out).mkdir(parents=True, exist_ok=True)
    vram = VramPeak(); vram.start()
    t0, thinking, no_conf = time.time(), 0, 0
    with open(Path(a.out) / f"predictions-{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            resp = post(a.url, request_body(build_messages(D / r["image"])))
            text, conf, think = parse_response(resp)
            lp = (resp["choices"][0].get("logprobs") or {}).get("content") or []
            thinking += think
            no_conf += conf is None
            p = parse_output(text)
            f.write(json.dumps({"id": r["id"], "raw": text, "valid": p.valid, "error": p.error,
                                **to_api(p, conf if conf is not None else 0.0), "conf_raw": conf,
                                "category_alts": value_alternatives(lp, "category"),
                                "type_alts": value_alternatives(lp, "incident_type")},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 50 == 0:
                print(i, len(rows), f"{(time.time() - t0) / (i + 1):.2f}s/img", text[:120], flush=True)
    vram.stop.set(); vram.join()
    stats = {"tag": a.tag, "n": len(rows), "s_per_img": round((time.time() - t0) / max(len(rows), 1), 3),
             "vram_peak_mib_total": vram.peak, "rows_with_thinking": thinking, "rows_without_confidence": no_conf}
    (Path(a.out) / f"stats-{a.tag}.json").write_text(json.dumps(stats, indent=2))
    print("done", json.dumps(stats))
    if thinking:
        sys.exit(f"{thinking} rows contain thinking output — enable_thinking=False not applied")


if __name__ == "__main__":
    main()
