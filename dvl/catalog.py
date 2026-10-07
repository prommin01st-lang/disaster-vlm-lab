"""ชุดประเภทเหตุเฟส 1 — key ตรงกับ SOS incident catalog (sosv1-backend library.json)."""
import re
from pathlib import Path

# key -> (category key, ชื่อไทย) — เฉพาะกลุ่ม A ที่ดูจากภาพได้และมีข้อมูล
PHASE1_TYPES: dict[str, tuple[str, str]] = {
    "forest_fire": ("fire_hazard", "ไฟป่า"),
    "building_fire": ("fire_hazard", "ไหม้อาคาร"),
    "vehicle_fire": ("fire_hazard", "ไหม้รถ"),
    "smoke_detected": ("fire_hazard", "พบควัน / กลิ่นไหม้"),
    "flood": ("disaster", "น้ำท่วม / น้ำป่าไหลหลาก"),
    "storm": ("disaster", "พายุ / ลมรุนแรง"),
    "landslide": ("disaster", "ดินถล่ม / โคลนถล่ม"),
    "damaged_structure": ("disaster", "อาคารเสียหาย / เสี่ยงถล่ม"),
    "road_hazard": ("disaster", "ถนนทรุด / หลุม / สิ่งกีดขวางอันตราย"),
    "building_collapse": ("rescue", "อาคารถล่ม"),
    "water_rescue": ("rescue", "คนติดค้างกลางน้ำ"),
    "vehicle_collision": ("accident", "รถชน"),
}

# unsure มีใน catalog จริง; no_incident เป็นค่าของโมเดลเท่านั้น (ไม่มีใน SOS)
EXTRA_TYPES: dict[str, tuple[str | None, str]] = {
    "unsure": ("other", "ไม่แน่ใจประเภทเหตุ"),
    "no_incident": (None, "ไม่ใช่เหตุ"),
}

ALLOWED_TYPES: dict[str, tuple[str | None, str]] = {**PHASE1_TYPES, **EXTRA_TYPES}

SEVERITIES: tuple[str, ...] = ("none", "mild", "severe")


def category_of(incident_type: str) -> str | None:
    return ALLOWED_TYPES[incident_type][0]


_SECTION = re.compile(r"^## \d+\. .*\(`([a-z_]+)`\)\s*$")
_ROW = re.compile(r"^\| `([a-z_]+)` \|")


def sos_catalog_keys(md_path: Path) -> dict[str, str]:
    """อ่าน incident-catalog.md ของ SOS → {type key: category key}"""
    out: dict[str, str] = {}
    category = None
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if m := _SECTION.match(line):
            category = m.group(1)
        elif category and (m := _ROW.match(line)):
            out[m.group(1)] = category
        elif line.startswith("## ") and not _SECTION.match(line):
            category = None  # ส่วน "ไอคอนกลุ่ม service" ไม่ใช่หมวดเหตุ
    return out
