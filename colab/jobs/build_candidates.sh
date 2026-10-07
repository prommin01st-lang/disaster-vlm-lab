set -e
python scripts/build_candidates.py
python - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("out/candidates/candidates.jsonl")]
print(collections.Counter(r["source"] for r in rows))
print(collections.Counter(len(r["candidates"]) == 1 for r in rows))
EOF
rm -rf out/candidates/images   # ภาพอยู่บน HF แล้ว ไม่ต้องดึงกลับ
