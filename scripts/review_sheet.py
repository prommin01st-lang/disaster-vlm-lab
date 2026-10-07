"""สร้างชีตให้คนตรวจ label ของ test (สุ่มไม่เกิน 25 ภาพ/คลาส) แล้ว apply กลับเป็น test_gold.jsonl
ใช้:  python scripts/review_sheet.py make   |   python scripts/review_sheet.py apply"""
import csv, html, json, random, sys, tarfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from dvl.catalog import ALLOWED_TYPES, category_of
from dvl.schema import target_json

D = Path("data/dataset_v1"); R = Path("data/review"); R.mkdir(parents=True, exist_ok=True)


def make():
    # Download images if not present
    if not (Path("data") / "dataset_v1" / "images").exists():
        repo = f"{HfApi().whoami()['name']}/dvl-data"
        tarfile.open(hf_hub_download(repo, "dataset_v1.tar", repo_type="dataset")).extractall("data", filter="data")

    # Load test set
    rows = [json.loads(l) for l in open(D / "test.jsonl", encoding="utf-8")]

    # Group by incident_type
    by = defaultdict(list)
    for r in rows:
        by[r["incident_type"]].append(r)

    # Sample up to 25 per class, sorted by type
    rng = random.Random(20261007)
    pick = [r for t in sorted(by) for r in rng.sample(by[t], min(25, len(by[t])))]

    # Count by label_source in picked set
    by_source = defaultdict(int)
    for r in pick:
        by_source[r["label_source"]] += 1

    # If fewer than 60 teacher-labeled rows, add more teacher rows
    teacher_count = by_source.get("teacher", 0)
    if teacher_count < 60:
        picked_ids = {r["id"] for r in pick}
        teacher_rows = [r for r in rows if r["label_source"] == "teacher" and r["id"] not in picked_ids]
        need = 60 - teacher_count
        to_add = rng.sample(teacher_rows, min(need, len(teacher_rows)))
        pick.extend(to_add)
        for r in to_add:
            by_source[r["label_source"]] += 1

    # Print counts
    print(f"Picked rows by label_source: {dict(by_source)}")

    # Write CSV
    with open(R / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "incident_type", "severity", "keep"])
        for r in pick:
            w.writerow([r["id"], r["incident_type"], r["severity"], "1"])

    # Build HTML with teacher highlighting and candidates
    cells = []
    for r in pick:
        img_path = html.escape(r["image"])
        row_id = html.escape(r["id"])
        incident_type = r["incident_type"]
        severity = r["severity"]
        source = r["source"]
        label_source = r["label_source"]

        # Build caption with candidates if they exist
        caption = f'<b>{row_id}</b><br>{incident_type} / {severity}<br><small>{source} · {label_source}</small>'
        if "candidates" in r and r["candidates"]:
            candidates_str = ", ".join(r["candidates"])
            caption += f'<br><small style="color:#999;">candidates: {html.escape(candidates_str)}</small>'

        # Apply CSS class for teacher rows
        teacher_class = ' class="teacher"' if label_source == "teacher" else ''
        cells.append(f'<figure{teacher_class}><img src="../dataset_v1/{img_path}" loading="lazy"><figcaption>{caption}</figcaption></figure>')

    cells_html = "".join(cells)
    types = " · ".join(f"<code>{k}</code>" for k in ALLOWED_TYPES)
    (R / "review.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>Test review</title><style>'
        f'body{{font-family:sans-serif;margin:16px}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}'
        f'img{{width:100%;height:180px;object-fit:cover}}figure{{margin:0}}figure.teacher{{border:2px solid #ff9800;padding:8px;border-radius:4px}}'
        f'figure.teacher figcaption small{{color:#ff9800;font-weight:bold}}</style>'
        f'<p>แก้ใน review.csv: เปลี่ยน incident_type/severity ถ้าผิด, keep=0 ถ้าภาพใช้ไม่ได้</p><p>{types}</p>'
        f'<main>{cells_html}</main>', encoding="utf-8")
    print(f"{len(pick)} rows → {R/'review.html'} + {R/'review.csv'}")


def apply():
    # Load test set keyed by id
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(D / "test.jsonl", encoding="utf-8")}
    out, bad = [], []

    # Track disagreements
    disagree_stats = {
        "total": 0,
        "by_source": defaultdict(lambda: {"total": 0, "changed_type": 0, "changed_severity": 0, "discarded": 0})
    }

    for c in csv.DictReader(open(R / "review.csv", encoding="utf-8")):
        original_id = c["id"]
        original_row = rows[original_id]
        original_source = original_row["label_source"]

        if c["keep"].strip() != "1":
            # Discarded row
            disagree_stats["by_source"][original_source]["total"] += 1
            disagree_stats["by_source"][original_source]["discarded"] += 1
            disagree_stats["total"] += 1
            continue

        t, s = c["incident_type"].strip(), c["severity"].strip()
        try:
            tgt = target_json(t, s)
        except (KeyError, ValueError) as e:
            bad.append((c["id"], str(e))); continue

        r = rows[c["id"]]

        # Track changes
        disagree_stats["by_source"][original_source]["total"] += 1
        if t != r["incident_type"]:
            disagree_stats["by_source"][original_source]["changed_type"] += 1
        if s != r["severity"]:
            disagree_stats["by_source"][original_source]["changed_severity"] += 1
        disagree_stats["total"] += 1

        out.append({**r, "incident_type": t, "category": category_of(t), "severity": s,
                    "target": tgt, "label_source": "human"})

    if bad:
        sys.exit(f"แก้แถวเหล่านี้ใน review.csv ก่อน: {bad}")

    # Print disagreement stats
    print("\n--- Disagreement Stats by Original label_source ---")
    for source in sorted(disagree_stats["by_source"].keys()):
        stats = disagree_stats["by_source"][source]
        total = stats["total"]
        if total > 0:
            changed_type = stats["changed_type"]
            changed_severity = stats["changed_severity"]
            discarded = stats["discarded"]
            print(f"\n{source} ({total} rows):")
            print(f"  changed_type: {changed_type} ({100*changed_type//total if total else 0}%)")
            print(f"  changed_severity: {changed_severity} ({100*changed_severity//total if total else 0}%)")
            print(f"  discarded: {discarded} ({100*discarded//total if total else 0}%)")

    # Write to JSON file
    stats_summary = {}
    for source in sorted(disagree_stats["by_source"].keys()):
        stats = disagree_stats["by_source"][source]
        total = stats["total"]
        stats_summary[source] = {
            "total": total,
            "changed_type": stats["changed_type"],
            "changed_type_pct": round(100 * stats["changed_type"] / total, 1) if total > 0 else 0,
            "changed_severity": stats["changed_severity"],
            "changed_severity_pct": round(100 * stats["changed_severity"] / total, 1) if total > 0 else 0,
            "discarded": stats["discarded"],
            "discarded_pct": round(100 * stats["discarded"] / total, 1) if total > 0 else 0,
        }

    (R / "review_stats.json").write_text(json.dumps(stats_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nStats saved to {R/'review_stats.json'}")

    # Write test_gold.jsonl
    with open(D / "test_gold.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"test_gold: {len(out)} rows")


{"make": make, "apply": apply}[sys.argv[1]]()
