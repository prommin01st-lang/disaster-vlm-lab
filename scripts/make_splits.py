"""dedupe ภาพเกือบซ้ำ → split แบบไม่รั่ว → balance เฉพาะ train → dataset_v1.tar ขึ้น HF"""
import json, os, tarfile
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

from dvl.catalog import category_of
from dvl.schema import target_json
from dvl.splits import balance, group_near_duplicates, stratified_split

SEED = 20261007
base = Path("data/candidates")
rows = [json.loads(l) for l in open(base / "labeled.jsonl", encoding="utf-8")]
groups = group_near_duplicates([r["ahash"] for r in rows], max_dist=4)
split = stratified_split([r["incident_type"] for r in rows], groups,
                         {"train": 0.80, "val": 0.05, "test": 0.15}, seed=SEED)
by = {"train": [], "val": [], "test": []}
for r, s in zip(rows, split):
    by[s].append({"id": r["id"], "image": r["image"], "incident_type": r["incident_type"],
                  "category": category_of(r["incident_type"]), "severity": r["severity"],
                  "target": target_json(r["incident_type"], r["severity"]),
                  "source": r["source"], "label_source": r["label_source"]})

# balance เฉพาะ train: cap ต่อคลาส 1200, no_incident ≤ 25% ของ train
train = balance(by["train"], "incident_type", cap=1200, seed=SEED)
n_inc = sum(r["incident_type"] != "no_incident" for r in train)
cap_neg = n_inc // 3  # no_incident / ทั้งหมด ≤ 25%
neg = [r for r in train if r["incident_type"] == "no_incident"][:cap_neg]
by["train"] = [r for r in train if r["incident_type"] != "no_incident"] + neg

out = Path("data/dataset_v1"); out.mkdir(parents=True, exist_ok=True)
for name, rs in by.items():
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rs), Counter(r["incident_type"] for r in rs).most_common())

used = {r["image"] for rs in by.values() for r in rs}
with tarfile.open("data/dataset_v1.tar", "w") as t:
    for name in by:
        t.add(out / f"{name}.jsonl", arcname=f"dataset_v1/{name}.jsonl")
    for img in sorted(used):
        t.add(base / img, arcname=f"dataset_v1/{img}")
api = HfApi(); repo = f"{api.whoami()['name']}/dvl-data"
api.upload_file(path_or_fileobj="data/dataset_v1.tar", path_in_repo="dataset_v1.tar",
                repo_id=repo, repo_type="dataset")
os.makedirs("out", exist_ok=True)
for name in by:  # ส่ง jsonl กลับเครื่องไว้ดู/ทำ review sheet
    os.system(f"cp {out}/{name}.jsonl out/")
print("uploaded dataset_v1.tar")
