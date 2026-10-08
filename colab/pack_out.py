import os, tarfile
from pathlib import Path

out = Path("/content/dvl/out")
if os.environ.get("PACK_MIN"):  # สำรองตอนดึงเต็มไม่ได้: เฉพาะไฟล์เล็กที่สำคัญ (ไม่เอา optimizer/ภาพ/predictions ใหญ่)
    keep = [p for p in out.rglob("*") if p.is_file() and (
        p.name in ("job.log", "log_history.json", "trainer_state.json")
        or p.name.startswith("adapter_"))]
    with tarfile.open("/content/out_min.tar.gz", "w:gz") as t:
        for p in keep:
            t.add(p, arcname=str("out" / p.relative_to(out)))
    print("packed min", len(keep), "files")
else:
    with tarfile.open("/content/out.tar.gz", "w:gz") as t:  # ต้องปิดไฟล์ให้ flush ก่อน download
        t.add(out, arcname="out")
