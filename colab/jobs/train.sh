set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/train.py
rm -rf out/ckpt   # checkpoint ใหญ่ ไม่ต้องดึงกลับ (adapter อยู่ใน out/adapter + HF) — ถ้าเทรนล้ม ckpt ยังอยู่ให้ fallback ดึง
