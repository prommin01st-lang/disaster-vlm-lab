"""python scripts/evaluate.py [--name SUFFIX] data/dataset_v1/test_gold.jsonl runs/.../predictions-base.jsonl [...]
รายงานเขียนที่ reports/<tag>[-SUFFIX].json (รวมแยกตาม label_source ถ้า gold มี)"""
import json, sys
from pathlib import Path

from dvl.metrics import compute_metrics

args = sys.argv[1:]
suffix = ""
if "--name" in args:
    i = args.index("--name")
    suffix = "-" + args[i + 1]
    del args[i:i + 2]
gold = {json.loads(l)["id"]: json.loads(l) for l in open(args[0], encoding="utf-8")}
Path("reports").mkdir(exist_ok=True)
results = {}
for path in args[1:]:
    pred = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}
    ids = [i for i in gold if i in pred]
    if len(ids) != len(gold):
        print(f"WARN {path}: {len(gold) - len(ids)} gold rows missing in predictions")
    tag = Path(path).stem.removeprefix("predictions-")
    results[tag] = m = compute_metrics([gold[i] for i in ids], [pred[i] for i in ids])
    groups = {}
    for ls in sorted({gold[i].get("label_source") for i in ids} - {None}):
        gi = [i for i in ids if gold[i].get("label_source") == ls]
        g = compute_metrics([gold[i] for i in gi], [pred[i] for i in gi])
        groups[ls] = {k: g[k] for k in ("n", "type_macro_f1", "type_accuracy", "false_alarm_rate", "miss_rate")}
    if groups:
        m["by_label_source"] = groups
    Path(f"reports/{tag}{suffix}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))

keys = ["n", "json_valid_rate", "type_accuracy", "type_macro_f1", "category_accuracy",
        "severity_accuracy", "false_alarm_rate", "miss_rate", "unsure_rate", "ece"]
print("| metric | " + " | ".join(results) + " |\n|---|" + "---|" * len(results))
for k in keys:
    print(f"| {k} | " + " | ".join(f"{results[t][k]:.3f}" if isinstance(results[t][k], float)
                                   else str(results[t][k]) for t in results) + " |")
print("\nper-type F1:")
for t in sorted(next(iter(results.values()))["per_type"]):
    print(f"  {t:20s} " + "  ".join(f"{results[x]['per_type'][t]['f1']:.2f}" for x in results)
          + f"   (n={next(iter(results.values()))['per_type'][t]['support']})")
lss = sorted({ls for r in results.values() for ls in r.get("by_label_source", {})})
if lss:
    print("\nby label_source:")
    print("| group | metric | " + " | ".join(results) + " |\n|---|---|" + "---|" * len(results))
    for ls in lss:
        for k in ("n", "type_macro_f1", "type_accuracy", "false_alarm_rate", "miss_rate"):
            print(f"| {ls} | {k} | " + " | ".join(
                (f"{v:.3f}" if isinstance(v := results[t]['by_label_source'][ls][k], float) else str(v))
                for t in results) + " |")
