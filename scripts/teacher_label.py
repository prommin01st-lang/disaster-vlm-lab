"""Qwen3.5-9B เลือก incident_type (ภายใน candidates) + severity ให้ภาพที่ label ยังไม่ตายตัว"""
import json, os, random, sys, tarfile, time
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

sys.path.insert(0, "scripts")
from vlm import generate_json, load_model  # noqa: E402

from dvl.prompt import build_teacher_messages  # noqa: E402
from dvl.schema import parse_output  # noqa: E402

TEACHER = os.environ.get("TEACHER", "Qwen/Qwen3.5-9B")
LIMIT = int(os.environ.get("LIMIT", "0"))  # >0 = รันแค่ N ภาพ (smoke)
TEACHER_MAX = int(os.environ.get("TEACHER_MAX", "0"))  # >0 = ย่อยแถว teacher (CrisisMMD) ให้เหลือรวมไม่เกิน N
SEED = 20261007
api = HfApi(); repo = f"{api.whoami()['name']}/dvl-data"
tarfile.open(hf_hub_download(repo, "candidates.tar", repo_type="dataset")).extractall("data")
base = Path("data/candidates")
rows = [json.loads(l) for l in open(base / "candidates.jsonl", encoding="utf-8")]
multi = [r for r in rows if len(r["candidates"]) > 1]
fixed_nosev = [r for r in rows if len(r["candidates"]) == 1 and r["severity_hint"] is None
               and r["candidates"][0] != "no_incident"]
print("pool", len(rows), "multi", len(multi), "fixed_need_severity", len(fixed_nosev),
      Counter(r["source"] for r in multi), flush=True)
if LIMIT:
    rows = multi[:LIMIT]
elif TEACHER_MAX and len(multi) + len(fixed_nosev) > TEACHER_MAX:
    # เก็บแถว teacher ที่ไม่ใช่ CrisisMMD ทั้งหมด + แถว severity ตายตัวทั้งหมด; สุ่มย่อย CrisisMMD แบบ deterministic
    keep_multi = [r for r in multi if r["source"] != "crisismmd"]
    cm = sorted((r for r in multi if r["source"] == "crisismmd"), key=lambda r: r["id"])
    room = max(0, TEACHER_MAX - len(fixed_nosev) - len(keep_multi))
    random.Random(SEED).shuffle(cm)
    drop_ids = {r["id"] for r in cm[room:]}
    rows = [r for r in rows if r["id"] not in drop_ids]
    print("subsampled crisismmd teacher rows:", room, "of", len(cm), "dropped", len(drop_ids), flush=True)

t0 = time.time()
model, processor = load_model(TEACHER)
print(f"model load {time.time() - t0:.0f}s", flush=True)
os.makedirs("out", exist_ok=True)
part = open("out/labeled_partial.jsonl", "w", encoding="utf-8")  # เขียนทีละแถว เผื่อ job ตายกลางทาง
out, dropped = [], {"outside": 0, "lowconf": 0, "invalid": 0}
n_teacher, t1 = 0, time.time()


def keep(rec):
    out.append(rec)
    part.write(json.dumps(rec, ensure_ascii=False) + "\n"); part.flush()


for i, r in enumerate(rows):
    if len(r["candidates"]) == 1:
        t = r["candidates"][0]
        sev = r["severity_hint"]
        if sev is None and t != "no_incident":  # type ตายตัวแต่ไม่มี severity → ให้ teacher ตัดสิน severity
            text, _ = generate_json(model, processor, build_teacher_messages(
                Image.open(base / r["image"]).convert("RGB"), (t,)))
            n_teacher += 1
            p = parse_output(text)
            sev = p.severity if p.valid and p.incident_type == t else "mild"
        if t == "no_incident":
            sev = "none"
        elif sev in (None, "none"):
            sev = "mild"
        keep({**r, "incident_type": t, "severity": sev, "label_source": "fixed", "teacher_conf": None})
        continue
    text, conf = generate_json(model, processor, build_teacher_messages(
        Image.open(base / r["image"]).convert("RGB"), tuple(r["candidates"])))
    n_teacher += 1
    p = parse_output(text)
    if LIMIT and n_teacher <= 8:
        print("RAW", repr(text), "conf", conf, "cands", r["candidates"], flush=True)
    allowed = set(r["candidates"]) | {"unsure", "no_incident"}
    if not p.valid:
        dropped["invalid"] += 1
    elif p.incident_type not in allowed:
        dropped["outside"] += 1
    elif conf is None or conf < 0.5:
        dropped["lowconf"] += 1
    else:
        sev = r["severity_hint"] if (r["severity_hint"] and p.incident_type != "no_incident") else p.severity
        if p.incident_type == "unsure" and sev == "none":
            sev = "mild"
        keep({**r, "incident_type": p.incident_type, "severity": sev,
              "label_source": "teacher", "teacher_conf": conf})
    if n_teacher and (n_teacher % 200 == 0 or (LIMIT and n_teacher % 10 == 0)):
        el = time.time() - t1
        print(f"{i + 1}/{len(rows)} teacher_calls={n_teacher} {el / n_teacher:.2f}s/img "
              f"kept={len(out)} {dropped}", flush=True)

part.close()
el = time.time() - t1
print(f"teacher_calls {n_teacher} total {el:.0f}s = {el / max(1, n_teacher):.2f}s/img")
with open(base / "labeled.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
tl = [r for r in out if r["label_source"] == "teacher"]
print("labeled", len(out), "dropped", dropped, "teacher_rows", len(tl),
      "unsure", sum(r["incident_type"] == "unsure" for r in tl))
print("teacher types", Counter(r["incident_type"] for r in tl).most_common())
