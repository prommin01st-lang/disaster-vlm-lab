"""รันโมเดล (base หรือ +adapter) บน test_gold/test → predictions-<tag>.jsonl
env: MODEL (default Qwen/Qwen3.5-2B), ADAPTER (HF repo/path หรือว่าง), TAG, LIMIT,
     SPLIT (gold = test_gold.jsonl จาก HF [default] | test = data/dataset_v1/test.jsonl)"""
import json, os, sys, tarfile, time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

sys.path.insert(0, "scripts")
from vlm import generate_json, load_model  # noqa: E402

from dvl.prompt import build_messages  # noqa: E402
from dvl.schema import parse_output, to_api  # noqa: E402

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
ADAPTER = os.environ.get("ADAPTER") or None
TAG = os.environ.get("TAG", "base")
LIMIT = int(os.environ.get("LIMIT", "0"))
SPLIT = os.environ.get("SPLIT", "gold")
repo = f"{HfApi().whoami()['name']}/dvl-data"
tarfile.open(hf_hub_download(repo, "dataset_v1.tar", repo_type="dataset")).extractall("data", filter="data")
if SPLIT == "gold":
    src = hf_hub_download(repo, "test_gold.jsonl", repo_type="dataset")
elif SPLIT == "test":
    src = "data/dataset_v1/test.jsonl"
else:
    raise SystemExit(f"bad SPLIT={SPLIT}")
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
rows = rows[:LIMIT] if LIMIT else rows

model, processor = load_model(MODEL, ADAPTER)
os.makedirs("out", exist_ok=True)
t0 = time.time()
with open(f"out/predictions-{TAG}.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        text, conf = generate_json(model, processor, build_messages(Image.open(Path("data/dataset_v1") / r["image"])))
        p = parse_output(text)
        f.write(json.dumps({"id": r["id"], "raw": text, "valid": p.valid, "error": p.error,
                            **to_api(p, conf if conf is not None else 0.0)}, ensure_ascii=False) + "\n")
        f.flush()
        if i % 50 == 0:
            print(i, len(rows), f"{(time.time() - t0) / (i + 1):.2f}s/img", text[:120], flush=True)
print("done", TAG, SPLIT, len(rows), f"{(time.time() - t0) / max(len(rows), 1):.2f}s/img")
