"""Prompt ที่เดียวของโปรเจกต์ — train / predict / serve ต้อง import จากที่นี่"""
from dvl.catalog import ALLOWED_TYPES, EXTRA_TYPES, PHASE1_TYPES


def _type_lines(keys) -> str:
    return "\n".join(f"- `{k}` ({ALLOWED_TYPES[k][1]})" for k in keys)


SYSTEM_PROMPT = (
    "คุณคือระบบคัดแยกภาพแจ้งเหตุฉุกเฉิน ดูภาพแล้วตอบเป็น JSON บรรทัดเดียวเท่านั้น ห้ามมีข้อความอื่น\n"
    'รูปแบบ: {"category": ..., "incident_type": ..., "severity": ...}\n'
    "incident_type ต้องเป็นค่าใดค่าหนึ่งต่อไปนี้:\n"
    f"{_type_lines(list(PHASE1_TYPES) + list(EXTRA_TYPES))}\n"
    "กติกา:\n"
    "- ตอบตามสิ่งที่เห็นในภาพเท่านั้น ห้ามเดาสาเหตุที่มองไม่เห็น\n"
    "- ภาพเป็นเหตุฉุกเฉินแต่ไม่ตรงประเภทใดข้างบน หรือดูไม่ออก ให้ตอบ `unsure`\n"
    "- ภาพไม่ใช่เหตุฉุกเฉิน (คน อาหาร วิว กราฟิก แผนที่ ข้อความ) ให้ตอบ `no_incident`\n"
    "- category คือหมวดของ incident_type; no_incident ให้ category เป็น null\n"
    "- severity: `none` เฉพาะ no_incident, `mild` เสียหาย/อันตรายเล็กน้อย, `severe` รุนแรงหรือมีคนตกอยู่ในอันตราย"
)

USER_INSTRUCTION = "ภาพนี้เป็นเหตุอะไร ตอบเป็น JSON"


def build_messages(image) -> list[dict]:
    img = {"type": "image"} if image is None else {"type": "image", "image": image}
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [img, {"type": "text", "text": USER_INSTRUCTION}]},
    ]


def build_teacher_messages(image, candidates: tuple[str, ...]) -> list[dict]:
    keys = [k for k in candidates if k not in EXTRA_TYPES] + list(EXTRA_TYPES)
    text = (
        "ภาพนี้มาจากรายงานภัยพิบัติ เลือก incident_type ที่ตรงกับ 'สิ่งที่เห็นในภาพ' มากที่สุด "
        "จากตัวเลือกนี้เท่านั้น:\n"
        f"{_type_lines(keys)}\n"
        "ถ้าภาพเป็นกราฟิก แผนที่ ข้อความ หรือคน/สิ่งของทั่วไปที่ไม่ใช่เหตุ ให้ตอบ `no_incident`\n"
        'ตอบ JSON บรรทัดเดียว: {"category": ..., "incident_type": ..., "severity": ...}'
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
    ]
