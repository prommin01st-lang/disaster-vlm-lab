# Disaster VLM Lab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** fine-tune VLM ขนาดเล็ก (Qwen3.5-2B) ให้ดูภาพแจ้งเหตุแล้วตอบ JSON ตามคลังประเภทเหตุของระบบ SOS (`category` / `incident_type` / `severity`) + `confidence` ที่คำนวณจาก logprob — แล้ววัดผลเทียบ base model อย่างเป็นระบบ

**Architecture:** แพ็กเกจ Python บริสุทธิ์ `dvl/` (catalog, prompt, schema, mapping, metrics — มี unit test ครบ รันบนเครื่อง) + สคริปต์งานหนัก `scripts/` ที่รันบน **Colab L4 ผ่าน Colab CLI** (สร้าง dataset, ให้ teacher Qwen3.5-9B ติด label, เทรน LoRA, predict) โดยมี `colab/run_job.sh` เป็นตัวส่งงาน/ดึงผลกลับ ข้อมูลกลางทางเก็บใน HF private dataset repo ข้ามเซสชัน

**Tech Stack:** Python 3.12, transformers 5.13.1, peft 0.19.1, trl 1.8.0, datasets, Pillow, pytest · Colab CLI 0.7.4 (บัญชี `บัญชีส่วนตัว`, 200 CU) · โมเดล `Qwen/Qwen3.5-2B` (student), `Qwen/Qwen3.5-9B` (teacher)

**Spec:** ส่วน "การตัดสินใจ (Spec)" ด้านล่าง — สรุปจากการคุยกับผู้ใช้ 2026-10-07 + `/home/petanque/Work/GitHubWork/SOS/docs/incident-catalog.md`

---

## การตัดสินใจ (Spec)

1. **ใช้ทดลอง/เรียนรู้เอง ไม่ขาย ไม่เผยแพร่ภาพ** → ใช้ dataset ไลเซนส์ NC ได้ (ถ้าจะขึ้นระบบ SOS จริงของ อบต. ต้องทบทวนไลเซนส์ใหม่)
2. **คลังประเภทเหตุ = SOS catalog** (7 หมวด · 68 ประเภท) ใช้ `key` เดียวกับ API
3. **เฟส 1 = เฉพาะประเภทที่ "ดูจากภาพได้ + มีข้อมูล" (กลุ่ม A) 12 ประเภท** ที่เหลือ (กลุ่ม B/C) โมเดลตอบ `unsure`
   - `earthquake` ย้ายไปกลุ่ม C: ภาพบอกได้แค่ "อาคารเสียหาย/ถล่ม" ไม่ใช่สาเหตุ → ภาพแผ่นดินไหวให้ label เป็น `damaged_structure` / `building_collapse`
4. **JSON ที่ระบบได้รับ:**
   ```json
   {"category": "disaster", "incident_type": "flood", "severity": "mild", "confidence": 0.91}
   ```
   - โมเดลสร้าง 3 ฟิลด์แรก (ลำดับ key ตายตัว) · `confidence` **server คำนวณจาก logprob** ของ token ค่า `incident_type` — โมเดลไม่ได้เขียนเลขเอง
   - ค่าพิเศษนอก catalog 1 ค่า: `no_incident` (ภาพไม่ใช่เหตุ เช่น เซลฟี่ อาหาร) → `category: null`, `severity: "none"`
   - `severity` = สิ่งที่เห็นในภาพ (`none` | `mild` | `severe`) **ไม่ใช่** ระดับ 2–4 ของ catalog
5. **โมเดล:** student `Qwen/Qwen3.5-2B` (Apache-2.0, non-thinking เป็นค่าเริ่มต้น, `tie_word_embeddings=True`) · Unsloth ไม่แนะนำ QLoRA กับ Qwen3.5 → **LoRA บน bf16 ที่ Colab L4** · teacher `Qwen/Qwen3.5-9B`
6. **Trust the eval:** ตัดสินด้วย macro-F1 ของ `incident_type` + false-alarm/miss rate บน test set ที่ **คนตรวจแล้ว** ไม่ใช่ loss

### ชุดประเภทเฟส 1

| incident_type | category | ชื่อ TH |
|---|---|---|
| `forest_fire` | `fire_hazard` | ไฟป่า |
| `building_fire` | `fire_hazard` | ไหม้อาคาร |
| `vehicle_fire` | `fire_hazard` | ไหม้รถ |
| `smoke_detected` | `fire_hazard` | พบควัน / กลิ่นไหม้ |
| `flood` | `disaster` | น้ำท่วม / น้ำป่าไหลหลาก |
| `storm` | `disaster` | พายุ / ลมรุนแรง |
| `landslide` | `disaster` | ดินถล่ม / โคลนถล่ม |
| `damaged_structure` | `disaster` | อาคารเสียหาย / เสี่ยงถล่ม |
| `road_hazard` | `disaster` | ถนนทรุด / หลุม / สิ่งกีดขวางอันตราย |
| `building_collapse` | `rescue` | อาคารถล่ม |
| `water_rescue` | `rescue` | คนติดค้างกลางน้ำ |
| `vehicle_collision` | `accident` | รถชน |
| `unsure` | `other` | ไม่แน่ใจประเภทเหตุ |
| `no_incident` | `null` | (นอก catalog) ไม่ใช่เหตุ |

### แหล่งข้อมูล (Hugging Face)

| Dataset | ใช้ทำอะไร | วิธี label |
|---|---|---|
| `QCRI/CrisisMMD` (configs `informative`, `damage`) | ภาพเหตุจริงจาก tweet 7 เหตุการณ์ | `event_name` → ชุดผู้สมัคร (candidates) → teacher เลือก · `damage` → severity |
| `anwan/DisasterVQA` | เติม fire/landslide/accident (ภาพจาก MEDIC+Incidents1M) | `disaster_type` → candidates → teacher เลือก (dedupe ภาพเพราะ 1 ภาพหลายคำถาม) |
| `AbdullahImran/balanced_wildfire_dataset` | `forest_fire` + severity | label ตายตัว |
| `fireviewer/fire_and_smoke_detection_very_hard_negative` (`existing_negative_fire_pics`) | hard negative (พระอาทิตย์ตก หมอก ไฟถนน) | `no_incident` ตายตัว |
| `hiennguyen9874/traffic-accident-detection` | `vehicle_collision` (มุม CCTV) | `is_accident` ตายตัว · cap 800/คลาส |
| `Arpitraj01/Pothole_classification` | `road_hazard` + severity | label ตายตัว |
| `nlphuji/flickr_1k_test_image_text_retrieval` | ภาพชีวิตประจำวัน → กัน false alarm | `no_incident` ตายตัว |

## Global Constraints

- Python ทุกตัวบนเครื่องรันผ่าน **`../iron-coach-th/.venv/bin/python`** (venv กลาง มี transformers 5.13.1 / peft 0.19.1 / trl 1.8.0 / datasets 5.0.0) — เพิ่มแค่ `pytest`
- บน Colab ปักเวอร์ชันเดียวกัน: `transformers==5.13.1 peft==0.19.1 trl==1.8.0` (ไฟล์ `requirements-colab.txt`)
- เครื่อง local = RTX 2060 **6GB** → งานโมเดลทั้งหมด (teacher/train/predict) รันบน **Colab L4** (`--gpu L4`) ไม่รันบนเครื่อง
- **ทุกงาน Colab ต้อง `colab stop` เสมอแม้ล้ม** (`trap ... EXIT`) — เซสชันค้าง = เผาเครดิต
- `SYSTEM_PROMPT` / `USER_INSTRUCTION` อยู่ที่เดียวใน `dvl/prompt.py` — train/predict/serve ต้อง import จากที่นี่เท่านั้น (บทเรียน iron-coach: prompt ไม่ตรง = พฤติกรรมเพี้ยน)
- ภาพทุกภาพย่อให้ด้านยาวสุด **≤ 512px** ตอนสร้าง dataset (คุมจำนวน image token + ขนาดข้อมูล)
- ลำดับ key ใน JSON เป้าหมายตายตัว: `category`, `incident_type`, `severity` (compact, `ensure_ascii=False`)
- generation: greedy (`do_sample=False`), `max_new_tokens=64`, ไม่เปิด thinking
- seed = `20261007` ทุกจุดที่สุ่ม
- โปรเจกต์เป็น git repo ของตัวเอง (เหมือนโปรเจกต์อื่นใน Model Trainer) · `data/`, `runs/`, `*.tar` อยู่ใน `.gitignore`
- ห้ามเก็บ HF token ใน repo — ส่งเข้า Colab ผ่าน `--env HF_TOKEN=...` จากตัวแปรสภาพแวดล้อมเครื่องเท่านั้น

## Review Focus

1. **โมเดลตอบ JSON พัง/มีข้อความเกิน/ถูกตัดกลางคัน** (เช่น ```` ```json ```` ครอบ, มีคำอธิบายต่อท้าย, ปีกกาไม่ปิด) → ต้องไม่ crash; คืน `valid=False` + fallback `unsure` และนับใน `json_valid_rate` — เทสต์ใน Task 2
2. **`incident_type` นอกชุดที่อนุญาต** (เช่น `earthquake`, คำไทย, ตัวพิมพ์ใหญ่ `Flood`) → ตัวพิมพ์ใหญ่ normalize ได้, ค่านอกชุด = invalid → `unsure` — เทสต์ใน Task 2
3. **`category` ขัดกับ `incident_type`** (`{"category":"accident","incident_type":"flood"}`) → เชื่อ `incident_type` แล้วเติม category จาก catalog; `no_incident` บังคับ severity `none` — เทสต์ใน Task 2
4. **ภาพแปลก ๆ:** RGBA/PNG โปร่งใส, grayscale, ภาพหมุนตาม EXIF, ภาพใหญ่มาก/เล็กมาก → แปลงเป็น RGB ≤512px ถูกทิศ ไม่ล้ม — เทสต์ใน Task 4
5. **ภาพซ้ำข้ามชุด train/test** (CrisisMMD มีรีทวีตภาพเดียวกันหลายรอบ, DisasterVQA 1 ภาพหลายคำถาม) → ภาพที่ hash ใกล้กันต้องอยู่ split เดียวกัน ไม่งั้นคะแนน test โกง — เทสต์ใน Task 4

---

## โครงสร้างไฟล์

```
disaster-vlm-lab/
├── PLAN.md                    # ไฟล์นี้
├── README.md                  # วิธีรัน + ผลลัพธ์ (เขียนใน Task 12)
├── .gitignore
├── requirements-colab.txt
├── dvl/                       # ตรรกะบริสุทธิ์ ไม่แตะ GPU — มี unit test
│   ├── __init__.py
│   ├── catalog.py             # ชุดประเภทเฟส 1 + category_of()
│   ├── schema.py              # target_json() / parse_output() → Prediction
│   ├── prompt.py              # SYSTEM_PROMPT, build_messages(), build_teacher_messages()
│   ├── confidence.py          # span_confidence() จาก token probs
│   ├── imgutil.py             # to_rgb_resized(), ahash(), hamming()
│   ├── mapping.py             # label ต้นทาง → Candidate
│   ├── splits.py              # group_near_duplicates(), stratified_split(), balance()
│   └── metrics.py             # compute_metrics()
├── tests/                     # pytest ของ dvl/*
├── scripts/                   # งานหนัก — รันบน Colab
│   ├── vlm.py                 # load_model(), generate_json() (ใช้ร่วม predict/teacher)
│   ├── build_candidates.py    # HF → ภาพ 512px + candidates.jsonl → HF repo
│   ├── teacher_label.py       # Qwen3.5-9B เลือก type/severity → labeled.jsonl
│   ├── make_splits.py         # dedupe + balance + split → train/val/test.jsonl
│   ├── review_sheet.py        # (local) สร้าง HTML/CSV ให้คนตรวจ test
│   ├── predict.py             # base หรือ +adapter → predictions.jsonl
│   ├── evaluate.py            # (local) predictions + gold → report
│   └── train.py               # LoRA SFT
├── colab/
│   ├── run_job.sh             # ส่งงานขึ้น Colab → poll → ดึงผล → stop
│   ├── bootstrap.py           # แตก bundle + pip install บน VM
│   ├── launch.py              # สั่งงานแบบ nohup เขียน log
│   ├── poll.py                # อ่าน log/สถานะ
│   └── jobs/smoke.sh          # งานทดสอบระบบส่งงาน
├── data/                      # (gitignored) ข้อมูลที่ดึงกลับมา
└── runs/                      # (gitignored) ผลแต่ละงาน
```

---

### Task 0: โครงโปรเจกต์ + catalog

**Files:**
- Create: `.gitignore`, `requirements-colab.txt`, `dvl/__init__.py`, `dvl/catalog.py`, `tests/test_catalog.py`

**Interfaces:**
- Produces: `PHASE1_TYPES: dict[str, tuple[str, str]]` (key → (category, ชื่อไทย)), `EXTRA_TYPES`, `ALLOWED_TYPES`, `SEVERITIES: tuple[str, ...]`, `category_of(incident_type: str) -> str | None`, `sos_catalog_keys(md_path: Path) -> dict[str, str]` (type key → category key)

- [ ] **Step 1: สร้างโฟลเดอร์ + git + pytest**

```bash
cd "/home/petanque/Work/GitHubWork/Model Trainer/disaster-vlm-lab"
git init
mkdir -p dvl tests scripts colab/jobs data runs
touch dvl/__init__.py
../iron-coach-th/.venv/bin/pip install pytest
cat > .gitignore <<'EOF'
data/
runs/
*.tar
*.tar.gz
__pycache__/
.pytest_cache/
EOF
cat > requirements-colab.txt <<'EOF'
transformers==5.13.1
peft==0.19.1
trl==1.8.0
datasets>=4.0
accelerate
huggingface_hub
pillow
EOF
```

- [ ] **Step 2: เขียนเทสต์ที่ล้มก่อน** — `tests/test_catalog.py`

```python
from pathlib import Path

import pytest

from dvl.catalog import (ALLOWED_TYPES, PHASE1_TYPES, SEVERITIES, category_of,
                         sos_catalog_keys)

SOS_MD = Path("/home/petanque/Work/GitHubWork/SOS/docs/incident-catalog.md")


def test_phase1_has_12_types():
    assert len(PHASE1_TYPES) == 12


def test_extra_types():
    assert category_of("unsure") == "other"
    assert category_of("no_incident") is None


def test_category_of_known():
    assert category_of("flood") == "disaster"
    assert category_of("building_collapse") == "rescue"
    assert category_of("vehicle_collision") == "accident"


def test_category_of_unknown_raises():
    with pytest.raises(KeyError):
        category_of("earthquake")


def test_severities():
    assert SEVERITIES == ("none", "mild", "severe")


def test_allowed_is_phase1_plus_extra():
    assert set(ALLOWED_TYPES) == set(PHASE1_TYPES) | {"unsure", "no_incident"}


@pytest.mark.skipif(not SOS_MD.exists(), reason="SOS repo not present")
def test_phase1_matches_sos_catalog():
    sos = sos_catalog_keys(SOS_MD)
    assert len(sos) == 68
    for key, (cat, _) in PHASE1_TYPES.items():
        assert sos[key] == cat, key
    assert sos["unsure"] == "other"
```

- [ ] **Step 3: รันให้เห็นว่าล้ม**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dvl.catalog'`

- [ ] **Step 4: เขียน `dvl/catalog.py`**

```python
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
```

- [ ] **Step 5: รันให้ผ่าน**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_catalog.py -v`
Expected: PASS 7 tests

- [ ] **Step 6: Commit**

```bash
git add .gitignore requirements-colab.txt dvl tests PLAN.md
git commit -m "feat: scaffold disaster-vlm-lab + phase-1 incident catalog"
```

---

### Task 1: Prompt (แหล่งเดียวของ prompt)

**Files:**
- Create: `dvl/prompt.py`, `tests/test_prompt.py`

**Interfaces:**
- Consumes: `PHASE1_TYPES`, `EXTRA_TYPES` จาก `dvl.catalog`
- Produces: `SYSTEM_PROMPT: str`, `USER_INSTRUCTION: str`, `build_messages(image) -> list[dict]` (รูปแบบ chat ของ transformers; `image` เป็น PIL หรือ `None` = ใส่ placeholder `{"type": "image"}` สำหรับเทรน), `build_teacher_messages(image, candidates: tuple[str, ...]) -> list[dict]`

- [ ] **Step 1: เขียนเทสต์ที่ล้มก่อน** — `tests/test_prompt.py`

```python
from dvl.catalog import ALLOWED_TYPES
from dvl.prompt import (SYSTEM_PROMPT, USER_INSTRUCTION, build_messages,
                        build_teacher_messages)


def test_system_prompt_lists_every_allowed_type():
    for key in ALLOWED_TYPES:
        assert f"`{key}`" in SYSTEM_PROMPT, key


def test_system_prompt_is_stable():
    # ถ้าเทสต์นี้พัง = prompt เปลี่ยน → ต้องเทรนใหม่ทั้งหมด (อัปเดต hash ด้วยความตั้งใจเท่านั้น)
    import hashlib
    h = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]
    assert h == SYSTEM_PROMPT_HASH


def test_build_messages_shape():
    msgs = build_messages("IMG")
    assert msgs[0] == {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][0] == {"type": "image", "image": "IMG"}
    assert msgs[1]["content"][1] == {"type": "text", "text": USER_INSTRUCTION}


def test_build_messages_placeholder_for_training():
    assert build_messages(None)[1]["content"][0] == {"type": "image"}


def test_teacher_messages_restrict_to_candidates():
    msgs = build_teacher_messages("IMG", ("flood", "storm"))
    text = msgs[1]["content"][1]["text"]
    assert "`flood`" in text and "`storm`" in text
    assert "`forest_fire`" not in text
    assert "`no_incident`" in text and "`unsure`" in text  # ทางออกเสมอ


SYSTEM_PROMPT_HASH = "REPLACE_IN_STEP_4"
```

- [ ] **Step 2: รันให้เห็นว่าล้ม**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_prompt.py -v`
Expected: FAIL — `No module named 'dvl.prompt'`

- [ ] **Step 3: เขียน `dvl/prompt.py`**

```python
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
```

- [ ] **Step 4: ล็อก hash ของ prompt**

```bash
../iron-coach-th/.venv/bin/python -c "import hashlib; from dvl.prompt import SYSTEM_PROMPT; print(hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12])"
```
นำค่าที่ได้ไปแทน `REPLACE_IN_STEP_4` ใน `tests/test_prompt.py`

- [ ] **Step 5: รันให้ผ่าน**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_prompt.py -v`
Expected: PASS 5 tests

- [ ] **Step 6: Commit**

```bash
git add dvl/prompt.py tests/test_prompt.py
git commit -m "feat: single-source system/teacher prompts with stability hash"
```

---

### Task 2: Schema — สร้าง target และ parse คำตอบโมเดล

**Files:**
- Create: `dvl/schema.py`, `tests/test_schema.py`

**Interfaces:**
- Consumes: `ALLOWED_TYPES`, `SEVERITIES`, `category_of`
- Produces:
  - `@dataclass(frozen=True) Prediction(category: str | None, incident_type: str, severity: str, valid: bool, error: str | None)`
  - `target_json(incident_type: str, severity: str) -> str` — สตริงที่ใช้เป็นคำตอบตอนเทรน
  - `parse_output(text: str) -> Prediction` — ไม่ raise ทุกกรณี
  - `to_api(pred: Prediction, confidence: float) -> dict` — JSON ที่ส่งให้ระบบ SOS

- [ ] **Step 1: เขียนเทสต์ที่ล้มก่อน** — `tests/test_schema.py`

```python
import json

import pytest

from dvl.schema import Prediction, parse_output, target_json, to_api


def test_target_json_exact_format():
    assert target_json("flood", "mild") == '{"category":"disaster","incident_type":"flood","severity":"mild"}'


def test_target_json_no_incident():
    assert target_json("no_incident", "none") == '{"category":null,"incident_type":"no_incident","severity":"none"}'


def test_target_json_rejects_bad_combo():
    with pytest.raises(ValueError):
        target_json("flood", "none")
    with pytest.raises(ValueError):
        target_json("no_incident", "severe")
    with pytest.raises(KeyError):
        target_json("earthquake", "mild")


def test_parse_roundtrip():
    p = parse_output(target_json("forest_fire", "severe"))
    assert p == Prediction("fire_hazard", "forest_fire", "severe", True, None)


@pytest.mark.parametrize("text", [
    '```json\n{"category":"disaster","incident_type":"flood","severity":"mild"}\n```',
    'คำตอบ: {"category":"disaster","incident_type":"flood","severity":"mild"} เพราะน้ำสูง',
    '{"category": "disaster", "incident_type": "Flood", "severity": "MILD"}',
])
def test_parse_tolerates_noise_and_case(text):
    p = parse_output(text)
    assert p.valid and p.incident_type == "flood" and p.severity == "mild"


def test_parse_fixes_inconsistent_category():
    p = parse_output('{"category":"accident","incident_type":"flood","severity":"mild"}')
    assert p.valid and p.category == "disaster"


def test_parse_forces_none_severity_for_no_incident():
    p = parse_output('{"category":null,"incident_type":"no_incident","severity":"mild"}')
    assert p.valid and p.severity == "none"


def test_parse_incident_with_none_severity_becomes_mild():
    p = parse_output('{"category":"disaster","incident_type":"flood","severity":"none"}')
    assert p.valid and p.severity == "mild"


@pytest.mark.parametrize("text,err", [
    ('', "no_json"),
    ('ไม่ทราบ', "no_json"),
    ('{"category":"disaster","incident_type":"flood"', "bad_json"),
    ('{"category":"disaster","incident_type":"earthquake","severity":"mild"}', "unknown_type"),
    ('{"category":"disaster","incident_type":"น้ำท่วม","severity":"mild"}', "unknown_type"),
    ('{"category":"disaster","incident_type":"flood","severity":"extreme"}', "unknown_severity"),
    ('["flood"]', "no_json"),
])
def test_parse_invalid_falls_back_to_unsure(text, err):
    p = parse_output(text)
    assert p == Prediction("other", "unsure", "mild", False, err)


def test_to_api():
    p = parse_output(target_json("flood", "severe"))
    assert to_api(p, 0.912345) == {"category": "disaster", "incident_type": "flood",
                                   "severity": "severe", "confidence": 0.912}
    json.dumps(to_api(p, 0.5))  # serialize ได้
```

- [ ] **Step 2: รันให้เห็นว่าล้ม**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_schema.py -v`
Expected: FAIL — `No module named 'dvl.schema'`

- [ ] **Step 3: เขียน `dvl/schema.py`**

```python
"""JSON เป้าหมายของโมเดล + parser ที่ทนคำตอบเพี้ยน (ไม่ raise)"""
import json
from dataclasses import dataclass

from dvl.catalog import ALLOWED_TYPES, SEVERITIES, category_of


@dataclass(frozen=True)
class Prediction:
    category: str | None
    incident_type: str
    severity: str
    valid: bool
    error: str | None


def _check_combo(incident_type: str, severity: str) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity {severity!r}")
    if (incident_type == "no_incident") != (severity == "none"):
        raise ValueError(f"bad combo {incident_type}/{severity}")


def target_json(incident_type: str, severity: str) -> str:
    category = category_of(incident_type)  # KeyError ถ้าไม่รู้จัก
    _check_combo(incident_type, severity)
    return json.dumps(
        {"category": category, "incident_type": incident_type, "severity": severity},
        ensure_ascii=False, separators=(",", ":"),
    )


def _fallback(error: str) -> Prediction:
    return Prediction("other", "unsure", "mild", False, error)


def parse_output(text: str) -> Prediction:
    start = text.find("{")
    if start < 0:
        return _fallback("no_json")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return _fallback("bad_json")
    if not isinstance(obj, dict):
        return _fallback("no_json")
    t = str(obj.get("incident_type", "")).strip().lower()
    s = str(obj.get("severity", "")).strip().lower()
    if t not in ALLOWED_TYPES:
        return _fallback("unknown_type")
    if s not in SEVERITIES:
        return _fallback("unknown_severity")
    # ซ่อม severity ให้สอดคล้องกับ type (เชื่อ incident_type มากกว่า)
    if t == "no_incident":
        s = "none"
    elif s == "none":
        s = "mild"
    return Prediction(category_of(t), t, s, True, None)


def to_api(pred: Prediction, confidence: float) -> dict:
    return {"category": pred.category, "incident_type": pred.incident_type,
            "severity": pred.severity, "confidence": round(float(confidence), 3)}
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_schema.py -v`
Expected: PASS ทั้งหมด (19 tests รวม parametrize)

- [ ] **Step 5: Commit**

```bash
git add dvl/schema.py tests/test_schema.py
git commit -m "feat: target JSON builder + tolerant output parser"
```

---

### Task 3: Confidence จาก logprob

**Files:**
- Create: `dvl/confidence.py`, `tests/test_confidence.py`

**Interfaces:**
- Produces: `span_confidence(tokens: list[str], probs: list[float], key: str = "incident_type") -> float | None` — ผลคูณ prob ของ token ที่คาบเกี่ยวกับ "ค่า" ของ key นั้นในข้อความที่ต่อจาก tokens; หาไม่เจอ = `None`

- [ ] **Step 1: เขียนเทสต์ที่ล้มก่อน** — `tests/test_confidence.py`

```python
import math

from dvl.confidence import span_confidence

TOKS = ['{"', 'category', '":"', 'dis', 'aster', '","', 'incident', '_type', '":"',
        'fl', 'ood', '","', 'severity', '":"', 'mild', '"}']


def test_product_of_value_tokens():
    probs = [1.0] * len(TOKS)
    probs[9], probs[10] = 0.9, 0.5  # 'fl', 'ood'
    assert math.isclose(span_confidence(TOKS, probs), 0.45)


def test_ignores_other_fields():
    probs = [0.1] * len(TOKS)
    probs[9] = probs[10] = 1.0
    assert math.isclose(span_confidence(TOKS, probs), 1.0)


def test_token_straddling_quote_counts():
    toks = ['{"incident_type":"', 'flood"', '}']
    assert math.isclose(span_confidence(toks, [1.0, 0.8, 1.0]), 0.8)


def test_missing_key_returns_none():
    assert span_confidence(['ไม่รู้'], [0.9]) is None


def test_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        span_confidence(['a'], [0.1, 0.2])
```

- [ ] **Step 2: รันให้เห็นว่าล้ม**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_confidence.py -v`
Expected: FAIL — `No module named 'dvl.confidence'`

- [ ] **Step 3: เขียน `dvl/confidence.py`**

```python
"""confidence = ความน่าจะเป็นที่โมเดลให้กับ token ของค่า incident_type (ไม่ใช่เลขที่โมเดลเขียนเอง)"""
import math
import re


def span_confidence(tokens: list[str], probs: list[float], key: str = "incident_type") -> float | None:
    if len(tokens) != len(probs):
        raise ValueError("tokens/probs length mismatch")
    text = "".join(tokens)
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]*)"', text)
    if not m:
        return None
    lo, hi = m.span(1)
    pos, picked = 0, []
    for tok, p in zip(tokens, probs):
        start, end = pos, pos + len(tok)
        if start < hi and end > lo:  # token คาบเกี่ยวช่วงค่า
            picked.append(p)
        pos = end
    return math.prod(picked) if picked else None
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_confidence.py -v`
Expected: PASS 5 tests

- [ ] **Step 5: Commit**

```bash
git add dvl/confidence.py tests/test_confidence.py
git commit -m "feat: logprob-based confidence for incident_type span"
```

---

### Task 4: ภาพ — normalize, hash, dedupe, split, balance

**Files:**
- Create: `dvl/imgutil.py`, `dvl/splits.py`, `tests/test_imgutil.py`, `tests/test_splits.py`

**Interfaces:**
- Produces:
  - `to_rgb_resized(img: PIL.Image.Image, max_side: int = 512) -> PIL.Image.Image`
  - `ahash(img: PIL.Image.Image) -> int` (64-bit average hash), `hamming(a: int, b: int) -> int`
  - `group_near_duplicates(hashes: list[int], max_dist: int = 4) -> list[int]` — คืน group id ต่อแถว
  - `stratified_split(labels: list[str], groups: list[int], fractions: dict[str, float], seed: int) -> list[str]` — คืนชื่อ split ต่อแถว; ทั้ง group อยู่ split เดียวกัน
  - `balance(rows: list[dict], key: str, cap: int, seed: int) -> list[dict]` — สุ่มตัดคลาสที่เกิน cap

- [ ] **Step 1: เขียนเทสต์ที่ล้มก่อน** — `tests/test_imgutil.py`

```python
from PIL import Image

from dvl.imgutil import ahash, hamming, to_rgb_resized


def test_resize_keeps_aspect_and_limits_side():
    out = to_rgb_resized(Image.new("RGB", (2000, 1000)))
    assert out.size == (512, 256) and out.mode == "RGB"


def test_small_image_not_upscaled():
    assert to_rgb_resized(Image.new("RGB", (100, 80))).size == (100, 80)


def test_rgba_composited_on_white():
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))  # โปร่งใสทั้งภาพ
    out = to_rgb_resized(img)
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (255, 255, 255)


def test_grayscale_and_palette():
    assert to_rgb_resized(Image.new("L", (10, 10))).mode == "RGB"
    assert to_rgb_resized(Image.new("P", (10, 10))).mode == "RGB"


def test_exif_rotation_applied():
    img = Image.new("RGB", (40, 20))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation: rotate 90 CW
    img.info["exif"] = exif.tobytes()
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out = to_rgb_resized(Image.open(io.BytesIO(buf.getvalue())))
    assert out.size == (20, 40)


def test_ahash_near_duplicate():
    a = Image.linear_gradient("L").convert("RGB")
    b = a.resize((128, 128))
    c = a.rotate(180)
    assert hamming(ahash(a), ahash(b)) <= 4
    assert hamming(ahash(a), ahash(c)) > 20
```

- [ ] **Step 2: เขียนเทสต์ที่ล้มก่อน** — `tests/test_splits.py`

```python
from collections import Counter

from dvl.splits import balance, group_near_duplicates, stratified_split


def test_group_near_duplicates():
    hashes = [0b0, 0b1, 0xFFFF_FFFF_FFFF_FFFF, 0b11]
    g = group_near_duplicates(hashes, max_dist=2)
    assert g[0] == g[1] == g[3] and g[2] != g[0]


def test_split_keeps_groups_together():
    labels = ["a"] * 50 + ["b"] * 50
    groups = [i // 2 for i in range(100)]  # คู่ละ group
    split = stratified_split(labels, groups, {"train": 0.8, "test": 0.2}, seed=1)
    for i in range(0, 100, 2):
        assert split[i] == split[i + 1]


def test_split_stratified_fractions():
    labels = ["a"] * 100 + ["b"] * 100
    split = stratified_split(labels, list(range(200)), {"train": 0.8, "val": 0.05, "test": 0.15}, seed=1)
    for lab in "ab":
        c = Counter(s for s, l in zip(split, labels) if l == lab)
        assert c["test"] == 15 and c["val"] == 5 and c["train"] == 80


def test_split_deterministic():
    labels = ["a"] * 30
    f = {"train": 0.5, "test": 0.5}
    assert stratified_split(labels, list(range(30)), f, 7) == stratified_split(labels, list(range(30)), f, 7)


def test_balance_caps_each_class():
    rows = [{"t": "a"}] * 10 + [{"t": "b"}] * 3
    out = balance(rows, "t", cap=5, seed=1)
    assert Counter(r["t"] for r in out) == {"a": 5, "b": 3}
```

- [ ] **Step 3: รันให้เห็นว่าล้ม**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_imgutil.py tests/test_splits.py -v`
Expected: FAIL — `No module named 'dvl.imgutil'`

- [ ] **Step 4: เขียน `dvl/imgutil.py`**

```python
from PIL import Image, ImageOps


def to_rgb_resized(img: Image.Image, max_side: int = 512) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        img = bg
    else:
        img = img.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img


def ahash(img: Image.Image) -> int:
    small = img.convert("L").resize((8, 8), Image.BILINEAR)
    px = list(small.getdata())
    avg = sum(px) / 64
    return sum(1 << i for i, p in enumerate(px) if p > avg)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
```

- [ ] **Step 5: เขียน `dvl/splits.py`**

```python
import random
from collections import defaultdict

from dvl.imgutil import hamming


def group_near_duplicates(hashes: list[int], max_dist: int = 4) -> list[int]:
    """union-find แบบ O(n^2) ผ่าน bucket 16 บิตแรก 4 ชุด (LSH แบบง่าย) — ภาพซ้ำ/เกือบซ้ำได้ group เดียวกัน"""
    parent = list(range(len(hashes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for band in range(4):
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[(h >> (16 * band)) & 0xFFFF].append(i)
        for idx in buckets.values():
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if hamming(hashes[i], hashes[j]) <= max_dist:
                        parent[find(i)] = find(j)
    return [find(i) for i in range(len(hashes))]


def stratified_split(labels: list[str], groups: list[int], fractions: dict[str, float], seed: int) -> list[str]:
    """แบ่งตาม group (label ของ group = label ของสมาชิกตัวแรก) แยกตามคลาส"""
    rng = random.Random(seed)
    members: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    by_label: dict[str, list[int]] = defaultdict(list)
    for g, idx in members.items():
        by_label[labels[idx[0]]].append(g)
    names = list(fractions)
    out = [""] * len(labels)
    for lab in sorted(by_label):
        gs = sorted(by_label[lab])
        rng.shuffle(gs)
        total = sum(len(members[g]) for g in gs)
        # เติม split ที่เล็กที่สุดก่อน (ไม่ใช่ train) ให้ได้ตามสัดส่วน ที่เหลือเป็น split แรก
        targets = {n: round(total * fractions[n]) for n in names[1:]}
        filled = {n: 0 for n in names}
        for g in gs:
            size = len(members[g])
            dest = next((n for n in names[1:] if filled[n] + size <= targets[n]), names[0])
            filled[dest] += size
            for i in members[g]:
                out[i] = dest
    return out


def balance(rows: list[dict], key: str, cap: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r[key]].append(r)
    out = []
    for k in sorted(by):
        rs = by[k][:]
        rng.shuffle(rs)
        out.extend(rs[:cap])
    return out
```

- [ ] **Step 6: รันให้ผ่าน**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_imgutil.py tests/test_splits.py -v`
Expected: PASS 11 tests

- [ ] **Step 7: Commit**

```bash
git add dvl/imgutil.py dvl/splits.py tests/test_imgutil.py tests/test_splits.py
git commit -m "feat: image normalize/ahash + leak-free grouped stratified split"
```

---

### Task 5: Mapping label ต้นทาง → candidates

**Files:**
- Create: `dvl/mapping.py`, `tests/test_mapping.py`

**Interfaces:**
- Consumes: `PHASE1_TYPES`
- Produces:
  - `@dataclass(frozen=True) Candidate(source: str, source_id: str, candidates: tuple[str, ...], severity_hint: str | None)` — ถ้า `len(candidates) == 1` = label ตายตัว ไม่ต้องถาม teacher
  - `crisismmd(event_name: str, informative: str, damage: str | None) -> tuple[tuple[str, ...], str | None]`
  - `disastervqa(disaster_type: str) -> tuple[str, ...]`
  - `wildfire(label: str) -> tuple[tuple[str, ...], str | None]`
  - `traffic(is_accident: bool) -> tuple[str, ...]`
  - `pothole(label: str) -> tuple[tuple[str, ...], str | None]`
  - `ALWAYS = ("unsure", "no_incident")` — ต่อท้ายทุกชุดที่ส่งให้ teacher

- [ ] **Step 1: เขียนเทสต์ที่ล้มก่อน** — `tests/test_mapping.py`

```python
import pytest

from dvl.catalog import ALLOWED_TYPES
from dvl.mapping import crisismmd, disastervqa, pothole, traffic, wildfire


def test_crisismmd_not_informative_is_fixed_no_incident():
    assert crisismmd("hurricane_harvey", "not_informative", None) == (("no_incident",), "none")


def test_crisismmd_wildfire_candidates():
    cands, sev = crisismmd("california_wildfires", "informative", None)
    assert "forest_fire" in cands and "flood" not in cands and sev is None


def test_crisismmd_earthquake_maps_to_damage_not_earthquake():
    cands, _ = crisismmd("mexico_earthquake", "informative", "severe_damage")
    assert "earthquake" not in cands and "building_collapse" in cands


def test_crisismmd_damage_hint():
    assert crisismmd("srilanka_floods", "informative", "little_or_no_damage")[1] == "mild"
    assert crisismmd("srilanka_floods", "informative", "severe_damage")[1] == "severe"


def test_crisismmd_unknown_event_raises():
    with pytest.raises(KeyError):
        crisismmd("mars_quake", "informative", None)


@pytest.mark.parametrize("dt", ["flood", "accidents", "hurricane", "landslide", "fire", "storm",
                                "wildfire", "other_disasters", "other", "earthquake"])
def test_disastervqa_all_types_map_into_allowed(dt):
    cands = disastervqa(dt)
    assert cands and all(c in ALLOWED_TYPES for c in cands)


def test_wildfire_fixed():
    assert wildfire("nofire") == (("no_incident",), "none")
    assert wildfire("severe") == (("forest_fire",), "severe")
    assert wildfire("fire") == (("forest_fire",), None)


def test_traffic_and_pothole():
    assert traffic(True) == ("vehicle_collision",)
    assert traffic(False) == ("no_incident",)
    assert pothole("none") == (("no_incident",), "none")
    assert pothole("medium") == (("road_hazard",), "mild")
```

- [ ] **Step 2: รันให้เห็นว่าล้ม**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_mapping.py -v`
Expected: FAIL — `No module named 'dvl.mapping'`

- [ ] **Step 3: เขียน `dvl/mapping.py`**

```python
"""label ของแต่ละ dataset → ชุด incident_type ที่เป็นไปได้ (teacher เลือกภายในชุดนี้)"""
from dataclasses import dataclass

ALWAYS = ("unsure", "no_incident")


@dataclass(frozen=True)
class Candidate:
    source: str
    source_id: str
    candidates: tuple[str, ...]
    severity_hint: str | None


_HURRICANE = ("storm", "flood", "damaged_structure", "building_collapse", "water_rescue", "road_hazard")
_CRISISMMD_EVENT = {
    "hurricane_harvey": _HURRICANE,
    "hurricane_irma": _HURRICANE,
    "hurricane_maria": _HURRICANE,
    "california_wildfires": ("forest_fire", "building_fire", "smoke_detected", "damaged_structure"),
    "mexico_earthquake": ("damaged_structure", "building_collapse", "landslide", "road_hazard"),
    "iraq_iran_earthquake": ("damaged_structure", "building_collapse", "landslide", "road_hazard"),
    "srilanka_floods": ("flood", "landslide", "water_rescue", "damaged_structure"),
}
_DAMAGE = {"little_or_no_damage": "mild", "mild_damage": "mild", "severe_damage": "severe"}


def crisismmd(event_name: str, informative: str, damage: str | None):
    cands = _CRISISMMD_EVENT[event_name]
    if informative == "not_informative":
        return ("no_incident",), "none"
    return cands, _DAMAGE.get(damage) if damage else None


_DVQA = {
    "flood": ("flood", "water_rescue", "damaged_structure"),
    "fire": ("building_fire", "vehicle_fire", "forest_fire", "smoke_detected"),
    "wildfire": ("forest_fire", "smoke_detected"),
    "landslide": ("landslide", "road_hazard"),
    "earthquake": ("damaged_structure", "building_collapse"),
    "hurricane": ("storm", "flood", "damaged_structure"),
    "storm": ("storm", "flood", "damaged_structure"),
    "accidents": ("vehicle_collision", "vehicle_fire"),
    "other_disasters": ("damaged_structure", "unsure"),
    "other": ("unsure",),
}


def disastervqa(disaster_type: str) -> tuple[str, ...]:
    return _DVQA[disaster_type]


_WILDFIRE_SEV = {"mild": "mild", "moderate": "severe", "severe": "severe", "fire": None}


def wildfire(label: str):
    if label == "nofire":
        return ("no_incident",), "none"
    return ("forest_fire",), _WILDFIRE_SEV[label]


def traffic(is_accident: bool) -> tuple[str, ...]:
    return ("vehicle_collision",) if is_accident else ("no_incident",)


_POTHOLE_SEV = {"low": "mild", "medium": "mild", "severe": "severe"}


def pothole(label: str):
    if label == "none":
        return ("no_incident",), "none"
    return ("road_hazard",), _POTHOLE_SEV[label]
```

- [ ] **Step 4: รันให้ผ่าน**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_mapping.py -v`
Expected: PASS ทั้งหมด

- [ ] **Step 5: Commit**

```bash
git add dvl/mapping.py tests/test_mapping.py
git commit -m "feat: map source dataset labels to candidate incident types"
```

---

### Task 6: Metrics

**Files:**
- Create: `dvl/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Prediction` (เฉพาะฟิลด์ `incident_type`, `category`, `severity`, `valid`)
- Produces: `compute_metrics(gold: list[dict], pred: list[dict]) -> dict` — `gold`/`pred` เป็น dict ที่มี key `incident_type`, `category`, `severity` (pred มี `valid`, `confidence` ด้วย) คืน:
  `{"n", "json_valid_rate", "type_accuracy", "type_macro_f1", "category_accuracy", "severity_accuracy", "false_alarm_rate", "miss_rate", "unsure_rate", "ece", "per_type": {type: {"precision","recall","f1","support"}}}`
  - `false_alarm_rate` = สัดส่วนของภาพ gold=`no_incident` ที่ถูกทายเป็นเหตุ (ไม่ใช่ no_incident/unsure)
  - `miss_rate` = สัดส่วนของภาพ gold=เหตุ ที่ถูกทายเป็น `no_incident`
  - `ece` = expected calibration error 10 bins ของ `confidence` เทียบความถูกของ type

- [ ] **Step 1: เขียนเทสต์ที่ล้มก่อน** — `tests/test_metrics.py`

```python
import math

from dvl.metrics import compute_metrics


def g(t, c="disaster", s="mild"):
    return {"incident_type": t, "category": c, "severity": s}


def p(t, c="disaster", s="mild", valid=True, conf=1.0):
    return {"incident_type": t, "category": c, "severity": s, "valid": valid, "confidence": conf}


def test_perfect():
    gold = [g("flood"), g("no_incident", None, "none")]
    m = compute_metrics(gold, [p("flood"), p("no_incident", None, "none")])
    assert m["type_accuracy"] == 1 and m["type_macro_f1"] == 1
    assert m["false_alarm_rate"] == 0 and m["miss_rate"] == 0 and m["json_valid_rate"] == 1


def test_false_alarm_and_miss():
    gold = [g("no_incident", None, "none"), g("no_incident", None, "none"), g("flood"), g("storm")]
    pred = [p("flood"), p("unsure", "other"), p("no_incident", None, "none"), p("storm")]
    m = compute_metrics(gold, pred)
    assert m["false_alarm_rate"] == 0.5   # unsure ไม่นับเป็น false alarm
    assert m["miss_rate"] == 0.5
    assert m["unsure_rate"] == 0.25


def test_macro_f1_over_gold_classes():
    gold = [g("flood"), g("flood"), g("storm")]
    pred = [p("flood"), p("storm"), p("storm")]
    m = compute_metrics(gold, pred)
    # flood P=1 R=.5 F1=.667 ; storm P=.5 R=1 F1=.667
    assert math.isclose(m["type_macro_f1"], 2 / 3, rel_tol=1e-6)
    assert m["per_type"]["flood"]["support"] == 2


def test_json_valid_rate():
    m = compute_metrics([g("flood")] * 2, [p("flood"), p("unsure", "other", valid=False)])
    assert m["json_valid_rate"] == 0.5


def test_ece_perfectly_calibrated_is_zero():
    gold = [g("flood")] * 4
    pred = [p("flood", conf=1.0)] * 4
    assert compute_metrics(gold, pred)["ece"] == 0


def test_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_metrics([g("flood")], [])
```

- [ ] **Step 2: รันให้เห็นว่าล้ม**

Run: `../iron-coach-th/.venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `No module named 'dvl.metrics'`

- [ ] **Step 3: เขียน `dvl/metrics.py`**

```python
from collections import Counter

NOT_ALARM = {"no_incident", "unsure"}


def _ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    n, total = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - avg)
    return round(total, 6)


def compute_metrics(gold: list[dict], pred: list[dict]) -> dict:
    if len(gold) != len(pred) or not gold:
        raise ValueError("gold/pred must be same non-zero length")
    n = len(gold)
    gt = [x["incident_type"] for x in gold]
    pt = [x["incident_type"] for x in pred]
    correct = [a == b for a, b in zip(gt, pt)]

    per_type = {}
    for t in sorted(set(gt)):
        tp = sum(a == t and b == t for a, b in zip(gt, pt))
        fp = sum(a != t and b == t for a, b in zip(gt, pt))
        fn = sum(a == t and b != t for a, b in zip(gt, pt))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_type[t] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}

    neg = [i for i, t in enumerate(gt) if t == "no_incident"]
    pos = [i for i, t in enumerate(gt) if t != "no_incident"]
    return {
        "n": n,
        "json_valid_rate": sum(bool(x.get("valid", True)) for x in pred) / n,
        "type_accuracy": sum(correct) / n,
        "type_macro_f1": sum(v["f1"] for v in per_type.values()) / len(per_type),
        "category_accuracy": sum(a["category"] == b["category"] for a, b in zip(gold, pred)) / n,
        "severity_accuracy": sum(a["severity"] == b["severity"] for a, b in zip(gold, pred)) / n,
        "false_alarm_rate": sum(pt[i] not in NOT_ALARM for i in neg) / len(neg) if neg else 0.0,
        "miss_rate": sum(pt[i] == "no_incident" for i in pos) / len(pos) if pos else 0.0,
        "unsure_rate": Counter(pt)["unsure"] / n,
        "ece": _ece([float(x.get("confidence", 0.0)) for x in pred], correct),
        "per_type": per_type,
    }
```

- [ ] **Step 4: รันให้ผ่าน + รันเทสต์ทั้งหมด**

Run: `../iron-coach-th/.venv/bin/python -m pytest -v`
Expected: PASS ทุกไฟล์

- [ ] **Step 5: Commit**

```bash
git add dvl/metrics.py tests/test_metrics.py
git commit -m "feat: eval metrics (macro-F1, false alarm, miss, ECE)"
```

---

### Task 7: ระบบส่งงานขึ้น Colab

**Files:**
- Create: `colab/run_job.sh`, `colab/bootstrap.py`, `colab/launch.py`, `colab/poll.py`, `colab/pack_out.py`, `colab/jobs/smoke.sh`

**Interfaces:**
- Produces: คำสั่ง `colab/run_job.sh <job.sh> [GPU] [timeout_min]` — job ทุกตัวเป็น bash script ที่รันใน `/content/dvl` บน VM; ทุกอย่างที่ job เขียนลง `/content/dvl/out/` จะถูกดึงกลับมาที่ `runs/<job>-<timestamp>/`; ส่ง `HF_TOKEN` จาก env เครื่องให้อัตโนมัติถ้ามี
- **หมายเหตุ:** `colab exec` มี timeout ต่อครั้ง (ค่าเริ่มต้น 30s) จึงต้องสั่งงานแบบ background แล้ว poll

- [ ] **Step 1: เขียน `colab/bootstrap.py`** (รันใน kernel ของ VM)

```python
import subprocess, tarfile
tarfile.open("/content/bundle.tar.gz").extractall("/content/dvl")
r = subprocess.run(["pip", "install", "-q", "-r", "/content/dvl/requirements-colab.txt"],
                   capture_output=True, text=True)
print("pip rc", r.returncode, r.stderr[-2000:])
```

- [ ] **Step 2: เขียน `colab/launch.py`** (อ่านชื่อ job จาก env `JOB`)

```python
import os, subprocess
job = os.environ["JOB"]
os.makedirs("/content/dvl/out", exist_ok=True)
cmd = f"cd /content/dvl && (bash {job}; echo EXIT_CODE=$? >> out/job.log) >> out/job.log 2>&1 &"
subprocess.Popen(["bash", "-c", cmd], env=dict(os.environ, PYTHONPATH="/content/dvl"))
print("launched", job)
```

- [ ] **Step 3: เขียน `colab/poll.py`**

```python
import os
p = "/content/dvl/out/job.log"
lines = open(p, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(p) else []
print("\n".join(lines[-15:]))
print("__DONE__" if any(l.startswith("EXIT_CODE=") for l in lines) else "__RUNNING__")
```

และ `colab/pack_out.py`:

```python
import tarfile
tarfile.open("/content/out.tar.gz", "w:gz").add("/content/dvl/out", arcname="out")
```

- [ ] **Step 4: เขียน `colab/run_job.sh`**

```bash
#!/usr/bin/env bash
# ใช้: colab/run_job.sh colab/jobs/<job>.sh [GPU=L4] [TIMEOUT_MIN=240]
set -euo pipefail
JOB=$1; GPU=${2:-L4}; TIMEOUT_MIN=${3:-240}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME="dvl-$(basename "$JOB" .sh)-$(date +%m%d%H%M)"
OUT="$ROOT/runs/$NAME"; mkdir -p "$OUT"
BUNDLE=$(mktemp --suffix .tar.gz)

cleanup() { colab stop -s "$NAME" >/dev/null 2>&1 || true; rm -f "$BUNDLE"; echo "[run_job] stopped $NAME"; }
trap cleanup EXIT   # หยุด VM เสมอ แม้ Ctrl+C หรือ error

tar -czf "$BUNDLE" -C "$ROOT" dvl scripts colab requirements-colab.txt
GPU_ARGS=(); [ "$GPU" != "CPU" ] && GPU_ARGS=(--gpu "$GPU")
colab new -s "$NAME" "${GPU_ARGS[@]}"
colab upload -s "$NAME" "$BUNDLE" /content/bundle.tar.gz
colab exec -s "$NAME" -f "$ROOT/colab/bootstrap.py" --timeout 900
ENV_ARGS=(--env "JOB=$JOB")
for v in HF_TOKEN LIMIT TEACHER MODEL ADAPTER TAG MAX_STEPS EPOCHS LR; do
  [ -n "${!v:-}" ] && ENV_ARGS+=(--env "$v=${!v}")
done
colab exec -s "$NAME" -f "$ROOT/colab/launch.py" "${ENV_ARGS[@]}"

DEADLINE=$(( $(date +%s) + TIMEOUT_MIN * 60 ))
while :; do
  sleep 60
  STATUS=$(colab exec -s "$NAME" -f "$ROOT/colab/poll.py" --timeout 60 || echo "__POLL_FAIL__")
  echo "$STATUS" | tail -4
  echo "$STATUS" | grep -q "__DONE__" && break
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "[run_job] TIMEOUT"; break; }
done

colab exec -s "$NAME" -f "$ROOT/colab/pack_out.py" --timeout 600
colab download -s "$NAME" /content/out.tar.gz "$OUT/out.tar.gz"
tar -xzf "$OUT/out.tar.gz" -C "$OUT" && rm "$OUT/out.tar.gz"
echo "[run_job] results → $OUT"
grep EXIT_CODE "$OUT/out/job.log"
```

```bash
chmod +x colab/run_job.sh
```

- [ ] **Step 5: เขียน `colab/jobs/smoke.sh`**

```bash
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import transformers, peft, trl, torch; print(transformers.__version__, peft.__version__, trl.__version__, torch.cuda.is_available())"
python -c "from dvl.prompt import SYSTEM_PROMPT; print(len(SYSTEM_PROMPT))"
echo ok > out/smoke.txt
```

- [ ] **Step 6: รัน smoke บน T4 (ถูกที่สุด) เพื่อพิสูจน์ระบบส่งงาน**

Run: `colab/run_job.sh colab/jobs/smoke.sh T4 15`
Expected: log แสดง `Tesla T4`, `5.13.1 0.19.1 1.8.0 True`, `EXIT_CODE=0`, มีไฟล์ `runs/dvl-smoke-*/out/smoke.txt` และบรรทัดสุดท้าย `[run_job] stopped ...`

- [ ] **Step 7: ยืนยันว่าไม่มีเซสชันค้าง**

Run: `colab sessions`
Expected: `No active sessions found on server.`

- [ ] **Step 8: Commit**

```bash
git add colab
git commit -m "feat: Colab job runner (bundle → launch → poll → fetch → always stop)"
```

---

### Task 8: ตัวช่วยเรียก VLM + สร้าง candidates

**Files:**
- Create: `scripts/vlm.py`, `scripts/build_candidates.py`, `colab/jobs/build_candidates.sh`

**Interfaces:**
- Consumes: `build_messages`, `build_teacher_messages`, `span_confidence`, `to_rgb_resized`, `ahash`, mapping functions
- Produces:
  - `scripts/vlm.py`: `load_model(model_id: str, adapter: str | None = None) -> tuple[model, processor]`, `generate_json(model, processor, messages: list[dict]) -> tuple[str, float | None]` (ข้อความดิบ, confidence)
  - ไฟล์ `out/candidates/candidates.jsonl` — 1 แถวต่อภาพ: `{"id": str, "source": str, "source_id": str, "image": "images/<id>.jpg", "candidates": [..], "severity_hint": str|null, "ahash": int}` + โฟลเดอร์ `images/` (JPEG q90, ≤512px)
  - HF private dataset repo `<HF_USER>/dvl-data` ไฟล์ `candidates.tar`

- [ ] **Step 1: เขียน `scripts/vlm.py`**

```python
"""โหลด VLM + generate แบบ greedy พร้อม confidence — ใช้ร่วมกันทั้ง teacher และ predict"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from dvl.confidence import span_confidence


def load_model(model_id: str, adapter: str | None = None):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


@torch.inference_mode()
def generate_json(model, processor, messages: list[dict]) -> tuple[str, float | None]:
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                         output_scores=True, return_dict_in_generate=True)
    new_ids = out.sequences[0, inputs["input_ids"].shape[1]:]
    probs = [torch.softmax(s[0].float(), -1)[t].item() for s, t in zip(out.scores, new_ids)]
    tok = processor.tokenizer
    pieces = [tok.decode([t], skip_special_tokens=False) for t in new_ids]
    keep = [i for i, t in enumerate(new_ids) if t not in tok.all_special_ids]
    text = "".join(pieces[i] for i in keep)
    return text, span_confidence([pieces[i] for i in keep], [probs[i] for i in keep])
```

- [ ] **Step 2: เขียน `scripts/build_candidates.py`**

```python
"""ดึง dataset จาก HF → ภาพ 512px + candidates.jsonl → อัปขึ้น HF private dataset repo"""
import hashlib, json, os, sys, tarfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

from dvl import mapping
from dvl.imgutil import ahash, to_rgb_resized

OUT = Path("out/candidates"); IMG = OUT / "images"
IMG.mkdir(parents=True, exist_ok=True)
rows, seen = [], set()


def add(source: str, source_id, image, cands, sev):
    rid = hashlib.sha1(f"{source}:{source_id}".encode()).hexdigest()[:16]
    if rid in seen or image is None:
        return
    try:
        img = to_rgb_resized(image)
    except Exception as e:  # ภาพเสีย — ข้ามแต่ log ไว้
        print("skip", source, source_id, e); return
    seen.add(rid)
    img.save(IMG / f"{rid}.jpg", quality=90)
    rows.append({"id": rid, "source": source, "source_id": str(source_id), "image": f"images/{rid}.jpg",
                 "candidates": list(cands), "severity_hint": sev, "ahash": ahash(img)})


# 1) CrisisMMD — join informative + damage ด้วย image_id
damage = {}
for split in ("train", "dev", "test"):
    for r in load_dataset("QCRI/CrisisMMD", "damage", split=split).remove_columns("image"):
        damage[r["image_id"]] = r["label"]
dmg_names = ["little_or_no_damage", "mild_damage", "severe_damage"]
for split in ("train", "dev", "test"):
    ds = load_dataset("QCRI/CrisisMMD", "informative", split=split)
    names = ds.features["label"].names
    for r in ds:
        d = damage.get(r["image_id"])
        cands, sev = mapping.crisismmd(r["event_name"], names[r["label"]], dmg_names[d] if d is not None else None)
        add("crisismmd", r["image_id"], r["image"], cands, sev)
print("crisismmd", len(rows))

# 2) DisasterVQA — 1 ภาพหลายคำถาม: key ด้วย image_id (หรือ hash ภาพถ้า id ว่าง)
for r in load_dataset("anwan/DisasterVQA", split="train"):
    sid = r["image_id"] or hashlib.sha1(r["image"].tobytes()).hexdigest()
    add("disastervqa", sid, r["image"], mapping.disastervqa(r["disaster_type"]), None)
print("+disastervqa", len(rows))

# 3) wildfire
for split in ("train", "validation", "test"):
    ds = load_dataset("AbdullahImran/balanced_wildfire_dataset", split=split)
    names = ds.features["label"].names
    for i, r in enumerate(ds):
        add("wildfire", f"{split}-{i}", r["image"], *mapping.wildfire(names[r["label"]]))
print("+wildfire", len(rows))

# 4) hard negatives
for i, r in enumerate(load_dataset("fireviewer/fire_and_smoke_detection_very_hard_negative",
                                   "existing_negative_fire_pics", split="train")):
    add("hardneg", i, r["image"], ("no_incident",), "none")

# 5) traffic accidents — cap 800 ต่อคลาส (มุม CCTV ซ้ำกันเยอะ)
cnt = {True: 0, False: 0}
for r in load_dataset("hiennguyen9874/traffic-accident-detection", split="train"):
    acc = bool(r["is_accident"])
    if cnt[acc] >= 800:
        continue
    cnt[acc] += 1
    add("traffic", r["image_id"], r["image"], mapping.traffic(acc), "none" if not acc else None)

# 6) pothole
for split in ("train", "validation", "test"):
    ds = load_dataset("Arpitraj01/Pothole_classification", split=split)
    names = ds.features["label"].names
    for i, r in enumerate(ds):
        add("pothole", f"{split}-{i}", r["image"], *mapping.pothole(names[r["label"]]))

# 7) ภาพชีวิตประจำวัน
for i, r in enumerate(load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test")):
    add("flickr", i, r["image"], ("no_incident",), "none")
print("total", len(rows))

with open(OUT / "candidates.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

if os.environ.get("HF_TOKEN"):
    with tarfile.open("out/candidates.tar", "w") as t:
        t.add(OUT, arcname="candidates")
    api = HfApi()
    repo = f"{api.whoami()['name']}/dvl-data"
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj="out/candidates.tar", path_in_repo="candidates.tar",
                    repo_id=repo, repo_type="dataset")
    os.remove("out/candidates.tar")
    print("uploaded →", repo)
else:
    sys.exit("HF_TOKEN missing — ไม่ได้อัปโหลด")
```

- [ ] **Step 3: เขียน `colab/jobs/build_candidates.sh`**

```bash
set -e
python scripts/build_candidates.py
python - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("out/candidates/candidates.jsonl")]
print(collections.Counter(r["source"] for r in rows))
print(collections.Counter(len(r["candidates"]) == 1 for r in rows))
EOF
rm -rf out/candidates/images   # ภาพอยู่บน HF แล้ว ไม่ต้องดึงกลับ
```

- [ ] **Step 4: เตรียม HF token บนเครื่อง**

Run: `../iron-coach-th/.venv/bin/hf auth whoami`
Expected: แสดงชื่อผู้ใช้ HF ถ้ายังไม่ล็อกอิน ให้ผู้ใช้รัน `! ../iron-coach-th/.venv/bin/hf auth login` แล้ว `export HF_TOKEN=$(cat ~/.cache/huggingface/token)` ก่อนส่งงาน

- [ ] **Step 5: รันบน CPU runtime (งานนี้ไม่ต้องใช้ GPU)**

Run: `HF_TOKEN=$(cat ~/.cache/huggingface/token) colab/run_job.sh colab/jobs/build_candidates.sh CPU 120`
Expected: `EXIT_CODE=0`, log มี `total` ราว 20k–30k แถว, `uploaded → <user>/dvl-data`, Counter ของ source ครบ 7 แหล่ง
ถ้า dataset ไหนโหลดไม่ได้ (ชื่อ split/คอลัมน์เปลี่ยน) ให้ดู error ใน `runs/*/out/job.log` แก้ชื่อตาม `https://datasets-server.huggingface.co/info?dataset=<id>` แล้วรันใหม่

- [ ] **Step 6: Commit**

```bash
git add scripts/vlm.py scripts/build_candidates.py colab/jobs/build_candidates.sh
git commit -m "feat: build candidate pool from 7 HF datasets"
```

---

### Task 9: Teacher ติด label + แบ่ง split

**Files:**
- Create: `scripts/teacher_label.py`, `scripts/make_splits.py`, `colab/jobs/teacher.sh`

**Interfaces:**
- Consumes: `load_model`, `generate_json`, `build_teacher_messages`, `parse_output`, `group_near_duplicates`, `stratified_split`, `balance`, `target_json`
- Produces:
  - `labeled.jsonl` — แถวของ candidates + `{"incident_type", "severity", "label_source": "fixed"|"teacher", "teacher_conf": float|null}` (แถวที่ teacher ตอบนอก candidates หรือ conf < 0.5 ถูกทิ้ง)
  - `train.jsonl`, `val.jsonl`, `test.jsonl` — `{"id", "image", "incident_type", "category", "severity", "target": target_json(...), "source", "label_source"}`
  - อัปขึ้น HF repo `dvl-data` เป็น `dataset_v1.tar` (มีภาพ + 3 ไฟล์ split)

- [ ] **Step 1: เขียน `scripts/teacher_label.py`**

```python
"""Qwen3.5-9B เลือก incident_type (ภายใน candidates) + severity ให้ภาพที่ label ยังไม่ตายตัว"""
import json, os, sys, tarfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

sys.path.insert(0, "scripts")
from vlm import generate_json, load_model  # noqa: E402

from dvl.prompt import build_teacher_messages  # noqa: E402
from dvl.schema import parse_output  # noqa: E402

TEACHER = os.environ.get("TEACHER", "Qwen/Qwen3.5-9B")
LIMIT = int(os.environ.get("LIMIT", "0"))  # >0 = รันแค่ N ภาพ (smoke)
api = HfApi(); repo = f"{api.whoami()['name']}/dvl-data"
tarfile.open(hf_hub_download(repo, "candidates.tar", repo_type="dataset")).extractall("data")
base = Path("data/candidates")
rows = [json.loads(l) for l in open(base / "candidates.jsonl", encoding="utf-8")]
if LIMIT:
    rows = [r for r in rows if len(r["candidates"]) > 1][:LIMIT]

model, processor = load_model(TEACHER)
out, dropped = [], {"outside": 0, "lowconf": 0, "invalid": 0}
for i, r in enumerate(rows):
    if len(r["candidates"]) == 1:
        t = r["candidates"][0]
        sev = r["severity_hint"]
        if sev is None:  # type ตายตัวแต่ไม่มี severity → ให้ teacher ตัดสิน severity
            text, _ = generate_json(model, processor, build_teacher_messages(
                Image.open(base / r["image"]), (t,)))
            p = parse_output(text)
            sev = p.severity if p.valid and p.incident_type == t else "mild"
        if t == "no_incident":
            sev = "none"
        out.append({**r, "incident_type": t, "severity": sev, "label_source": "fixed", "teacher_conf": None})
        continue
    text, conf = generate_json(model, processor, build_teacher_messages(
        Image.open(base / r["image"]), tuple(r["candidates"])))
    p = parse_output(text)
    allowed = set(r["candidates"]) | {"unsure", "no_incident"}
    if not p.valid:
        dropped["invalid"] += 1; continue
    if p.incident_type not in allowed:
        dropped["outside"] += 1; continue
    if conf is None or conf < 0.5:
        dropped["lowconf"] += 1; continue
    sev = r["severity_hint"] if (r["severity_hint"] and p.incident_type != "no_incident") else p.severity
    out.append({**r, "incident_type": p.incident_type, "severity": sev,
                "label_source": "teacher", "teacher_conf": conf})
    if i % 200 == 0:
        print(i, len(rows), dropped, flush=True)

os.makedirs("out", exist_ok=True)
with open(base / "labeled.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("labeled", len(out), "dropped", dropped)
```

- [ ] **Step 2: เขียน `scripts/make_splits.py`**

```python
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
```

- [ ] **Step 3: เขียน `colab/jobs/teacher.sh`**

```bash
set -e
python scripts/teacher_label.py
python scripts/make_splits.py
```

- [ ] **Step 4: Smoke teacher 30 ภาพ (ตรวจว่า output parse ได้และ conf มีค่า)**

Run: `HF_TOKEN=$(cat ~/.cache/huggingface/token) LIMIT=30 colab/run_job.sh colab/jobs/teacher.sh L4 40`
Expected: `labeled` ≥ 20 จาก 30, `dropped` ส่วนใหญ่เป็น lowconf ไม่ใช่ invalid — ถ้า invalid เยอะ ให้เปิด `job.log` ดูข้อความดิบ แล้วแก้ prompt teacher ใน `dvl/prompt.py` (ไม่แตะ `SYSTEM_PROMPT`)
หมายเหตุ: smoke นี้ make_splits จะทำงานกับข้อมูล 30 แถวและอัป `dataset_v1.tar` ตัวเล็ก — รอบเต็มจะเขียนทับ

- [ ] **Step 5: รันเต็ม**

Run: `HF_TOKEN=$(cat ~/.cache/huggingface/token) colab/run_job.sh colab/jobs/teacher.sh L4 480`
Expected: `EXIT_CODE=0`; `runs/*/out/{train,val,test}.jsonl`; ทุกคลาสเฟส 1 มีใน test ≥ 30 ภาพ — คลาสที่ต่ำกว่านั้นจดไว้ใน README (เป็นข้อจำกัดข้อมูล ไม่ใช่บั๊ก) · ระหว่างรัน เช็กอัตราเผาเครดิตด้วย `colab usage`

- [ ] **Step 6: คัดลอกผลเข้า `data/` + Commit โค้ด**

```bash
mkdir -p data/dataset_v1 && cp runs/dvl-teacher-*/out/*.jsonl data/dataset_v1/
git add scripts/teacher_label.py scripts/make_splits.py colab/jobs/teacher.sh
git commit -m "feat: teacher labeling (Qwen3.5-9B) + leak-free balanced splits"
```

---

### Task 10: คนตรวจ test set (gold)

**Files:**
- Create: `scripts/review_sheet.py`

**Interfaces:**
- Consumes: `data/dataset_v1/test.jsonl`, ภาพจาก HF `dataset_v1.tar`
- Produces: `data/review/review.html` (ตาราง ภาพ + label) และ `data/review/review.csv` (`id,incident_type,severity,keep`) → ผู้ใช้แก้ CSV → `python scripts/review_sheet.py apply` เขียน `data/dataset_v1/test_gold.jsonl`

- [ ] **Step 1: เขียน `scripts/review_sheet.py`**

```python
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
    if not (Path("data") / "dataset_v1" / "images").exists():
        repo = f"{HfApi().whoami()['name']}/dvl-data"
        tarfile.open(hf_hub_download(repo, "dataset_v1.tar", repo_type="dataset")).extractall("data")
    rows = [json.loads(l) for l in open(D / "test.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    for r in rows:
        by[r["incident_type"]].append(r)
    rng = random.Random(20261007)
    pick = [r for t in sorted(by) for r in rng.sample(by[t], min(25, len(by[t])))]
    with open(R / "review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "incident_type", "severity", "keep"])
        for r in pick:
            w.writerow([r["id"], r["incident_type"], r["severity"], "1"])
    cells = "".join(
        f'<figure><img src="../dataset_v1/{r["image"]}" loading="lazy"><figcaption>'
        f'<b>{html.escape(r["id"])}</b><br>{r["incident_type"]} / {r["severity"]}<br>'
        f'<small>{r["source"]} · {r["label_source"]}</small></figcaption></figure>' for r in pick)
    types = " · ".join(f"<code>{k}</code>" for k in ALLOWED_TYPES)
    (R / "review.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>Test review</title><style>'
        f'body{{font-family:sans-serif;margin:16px}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}'
        f'img{{width:100%;height:180px;object-fit:cover}}figure{{margin:0}}</style>'
        f'<p>แก้ใน review.csv: เปลี่ยน incident_type/severity ถ้าผิด, keep=0 ถ้าภาพใช้ไม่ได้</p><p>{types}</p>'
        f'<main>{cells}</main>', encoding="utf-8")
    print(f"{len(pick)} rows → {R/'review.html'} + {R/'review.csv'}")


def apply():
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(D / "test.jsonl", encoding="utf-8")}
    out, bad = [], []
    for c in csv.DictReader(open(R / "review.csv", encoding="utf-8")):
        if c["keep"].strip() != "1":
            continue
        t, s = c["incident_type"].strip(), c["severity"].strip()
        try:
            tgt = target_json(t, s)
        except (KeyError, ValueError) as e:
            bad.append((c["id"], str(e))); continue
        r = rows[c["id"]]
        out.append({**r, "incident_type": t, "category": category_of(t), "severity": s,
                    "target": tgt, "label_source": "human"})
    if bad:
        sys.exit(f"แก้แถวเหล่านี้ใน review.csv ก่อน: {bad}")
    with open(D / "test_gold.jsonl", "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"test_gold: {len(out)} rows")


{"make": make, "apply": apply}[sys.argv[1]]()
```

- [ ] **Step 2: สร้างชีต**

Run: `../iron-coach-th/.venv/bin/python scripts/review_sheet.py make`
Expected: `~300 rows → data/review/review.html + data/review/review.csv`

- [ ] **Step 3: 👤 ผู้ใช้ตรวจ** — เปิด `data/review/review.html` ในเบราว์เซอร์ แก้ `review.csv` (เปลี่ยน type/severity ที่ผิด, `keep=0` ภาพที่ใช้ไม่ได้) — **ขั้นนี้ต้องรอผู้ใช้** · บันทึกอัตราที่ teacher label ผิดไว้ใน README (บอกคุณภาพของ label train ด้วย)

- [ ] **Step 4: Apply**

Run: `../iron-coach-th/.venv/bin/python scripts/review_sheet.py apply`
Expected: `test_gold: N rows` (N ≈ จำนวน keep=1) ไม่มี error · อัป `test_gold.jsonl` ขึ้น HF repo:
```bash
../iron-coach-th/.venv/bin/hf upload <HF_USER>/dvl-data data/dataset_v1/test_gold.jsonl test_gold.jsonl --repo-type dataset
```

- [ ] **Step 5: Commit**

```bash
git add scripts/review_sheet.py
git commit -m "feat: human review sheet for gold test set"
```

---

### Task 11: Predict + Evaluate + Baseline

**Files:**
- Create: `scripts/predict.py`, `scripts/evaluate.py`, `colab/jobs/predict.sh`

**Interfaces:**
- Consumes: `load_model`, `generate_json`, `build_messages`, `parse_output`, `to_api`, `compute_metrics`
- Produces:
  - `out/predictions-<tag>.jsonl` — `{"id", "raw", "valid", "error", "category", "incident_type", "severity", "confidence"}`
  - `scripts/evaluate.py <gold.jsonl> <pred.jsonl> [<pred2.jsonl> ...]` → พิมพ์ตารางเทียบ + เขียน `reports/<tag>.json`

- [ ] **Step 1: เขียน `scripts/predict.py`**

```python
"""รันโมเดล (base หรือ +adapter) บน test_gold → predictions-<tag>.jsonl
env: MODEL (default Qwen/Qwen3.5-2B), ADAPTER (HF repo/path หรือว่าง), TAG, LIMIT"""
import json, os, sys, tarfile
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
repo = f"{HfApi().whoami()['name']}/dvl-data"
tarfile.open(hf_hub_download(repo, "dataset_v1.tar", repo_type="dataset")).extractall("data")
gold = hf_hub_download(repo, "test_gold.jsonl", repo_type="dataset")
rows = [json.loads(l) for l in open(gold, encoding="utf-8")]
rows = rows[:LIMIT] if LIMIT else rows

model, processor = load_model(MODEL, ADAPTER)
os.makedirs("out", exist_ok=True)
with open(f"out/predictions-{TAG}.jsonl", "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        text, conf = generate_json(model, processor, build_messages(Image.open(Path("data/dataset_v1") / r["image"])))
        p = parse_output(text)
        f.write(json.dumps({"id": r["id"], "raw": text, "valid": p.valid, "error": p.error,
                            **to_api(p, conf if conf is not None else 0.0)}, ensure_ascii=False) + "\n")
        if i % 50 == 0:
            print(i, len(rows), text[:120], flush=True)
print("done", TAG)
```

- [ ] **Step 2: เขียน `scripts/evaluate.py`**

```python
"""python scripts/evaluate.py data/dataset_v1/test_gold.jsonl runs/.../predictions-base.jsonl [...]"""
import json, sys
from pathlib import Path

from dvl.metrics import compute_metrics

gold = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
Path("reports").mkdir(exist_ok=True)
results = {}
for path in sys.argv[2:]:
    pred = {json.loads(l)["id"]: json.loads(l) for l in open(path, encoding="utf-8")}
    ids = [i for i in gold if i in pred]
    if len(ids) != len(gold):
        print(f"WARN {path}: {len(gold) - len(ids)} gold rows missing in predictions")
    tag = Path(path).stem.removeprefix("predictions-")
    results[tag] = m = compute_metrics([gold[i] for i in ids], [pred[i] for i in ids])
    Path(f"reports/{tag}.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))

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
```

`reports/` เก็บใน git (ไฟล์ JSON เล็ก ใช้เทียบย้อนหลัง)

- [ ] **Step 3: เขียน `colab/jobs/predict.sh`**

```bash
set -e
python scripts/predict.py
```

- [ ] **Step 4: Smoke 10 ภาพ**

Run: `HF_TOKEN=$(cat ~/.cache/huggingface/token) LIMIT=10 TAG=smoke colab/run_job.sh colab/jobs/predict.sh L4 30`
Expected: `runs/*/out/predictions-smoke.jsonl` 10 แถว, `confidence` เป็นตัวเลข 0–1 ไม่ใช่ 0 ทุกแถว · ถ้า confidence = 0 ทุกแถว แปลว่า span หาไม่เจอ → ดู `raw` ว่าโมเดลเว้นวรรคหลัง `:` หรือไม่ (regex รองรับแล้ว) หรือใช้ key อื่น

- [ ] **Step 5: Baseline เต็ม (Qwen3.5-2B ไม่จูน)**

Run: `HF_TOKEN=$(cat ~/.cache/huggingface/token) TAG=base colab/run_job.sh colab/jobs/predict.sh L4 60`
แล้ว: `../iron-coach-th/.venv/bin/python scripts/evaluate.py data/dataset_v1/test_gold.jsonl runs/dvl-predict-*/out/predictions-base.jsonl`
Expected: ตาราง metric ของ base — จดตัวเลขไว้ใน README หัวข้อ "Baseline" (คาดว่า json_valid_rate สูงแต่ macro-F1 ต่ำ เพราะ base ไม่รู้กติกา `unsure`/`no_incident`)

- [ ] **Step 6: Commit**

```bash
git add scripts/predict.py scripts/evaluate.py colab/jobs/predict.sh reports/base.json
git commit -m "feat: predict/evaluate pipeline + Qwen3.5-2B baseline"
```

---

### Task 12: เทรน LoRA + เทียบผล

**Files:**
- Create: `scripts/train.py`, `colab/jobs/train.sh`, `README.md`

**Interfaces:**
- Consumes: `build_messages`, train/val jsonl จาก `dataset_v1.tar`
- Produces: adapter ที่ `out/adapter/` + อัปขึ้น HF private model repo `<HF_USER>/dvl-qwen35-2b-lora` · ใช้ต่อด้วย `ADAPTER=<HF_USER>/dvl-qwen35-2b-lora TAG=lora_v1` ใน predict

- [ ] **Step 1: เขียน `scripts/train.py`**

```python
"""LoRA SFT บน Qwen3.5-2B (bf16, Colab L4) — loss เฉพาะ token คำตอบ
env: MAX_STEPS (smoke), EPOCHS (default 2), LR (default 1e-4)"""
import json, os, tarfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from dvl.prompt import build_messages

MODEL = "Qwen/Qwen3.5-2B"
api = HfApi(); user = api.whoami()["name"]
tarfile.open(hf_hub_download(f"{user}/dvl-data", "dataset_v1.tar", repo_type="dataset")).extractall("data")
D = Path("data/dataset_v1")


def load(name):
    return [json.loads(l) for l in open(D / f"{name}.jsonl", encoding="utf-8")]


train_rows, val_rows = load("train"), load("val")
processor = AutoProcessor.from_pretrained(MODEL)
processor.tokenizer.padding_side = "right"
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")

# พิมพ์ชื่อโมดูล linear ของ language model ครั้งแรก เพื่อยืนยัน target_modules (Qwen3.5 มี linear-attention ปนกับ full-attention)
names = sorted({n.split(".")[-1] for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and "visual" not in n})
print("linear module names (non-vision):", names)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"))
model.print_trainable_parameters()


def collate(batch):
    images = [[Image.open(D / r["image"])] for r in batch]
    full = [build_messages(None) + [{"role": "assistant", "content": [{"type": "text", "text": r["target"]}]}]
            for r in batch]
    prompt_only = [build_messages(None) for _ in batch]
    enc = processor.apply_chat_template(full, images=images, tokenize=True, return_dict=True,
                                        return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    for i, (pm, im) in enumerate(zip(prompt_only, images)):
        plen = processor.apply_chat_template([pm], images=[im], tokenize=True, add_generation_prompt=True,
                                             return_dict=True, return_tensors="pt")["input_ids"].shape[1]
        labels[i, :plen] = -100
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


args = SFTConfig(
    output_dir="out/ckpt", per_device_train_batch_size=4, per_device_eval_batch_size=4,
    gradient_accumulation_steps=4, num_train_epochs=float(os.environ.get("EPOCHS", 2)),
    max_steps=int(os.environ.get("MAX_STEPS", -1)), learning_rate=float(os.environ.get("LR", 1e-4)),
    lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, fp16=False,
    gradient_checkpointing=True, logging_steps=10, eval_strategy="steps", eval_steps=100,
    save_strategy="steps", save_steps=100, save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="eval_loss", report_to="none", remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True}, seed=20261007)
from datasets import Dataset  # noqa: E402
trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(train_rows),
                     eval_dataset=Dataset.from_list(val_rows),
                     data_collator=collate, processing_class=processor,
                     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
trainer.train()
model.save_pretrained("out/adapter")
if int(os.environ.get("MAX_STEPS", -1)) < 0:
    repo = f"{user}/dvl-qwen35-2b-lora"
    api.create_repo(repo, private=True, exist_ok=True)
    api.upload_folder(folder_path="out/adapter", repo_id=repo)
    print("uploaded →", repo)
```

- [ ] **Step 2: เขียน `colab/jobs/train.sh`**

```bash
set -e
python scripts/train.py
rm -rf out/ckpt   # checkpoint ใหญ่ ไม่ต้องดึงกลับ (adapter อยู่ใน out/adapter + HF)
```

- [ ] **Step 3: Smoke 20 step**

Run: `HF_TOKEN=$(cat ~/.cache/huggingface/token) MAX_STEPS=20 colab/run_job.sh colab/jobs/train.sh L4 40`
Expected: log แสดง `linear module names` (ตรวจว่ามี `q_proj`… ถ้า Qwen3.5 ใช้ชื่ออื่นใน linear-attention layer เช่น `in_proj_qkvz`/`out_proj` ให้เพิ่มชื่อนั้นลง regex `target_modules`), `trainable params` ราว 0.5–1.5%, loss ลดลงภายใน 20 step, ไม่มี OOM, `EXIT_CODE=0`
ถ้า `apply_chat_template(..., images=...)` ไม่รับ argument `images` ใน transformers 5.13.1 ให้เปลี่ยนเป็นใส่ภาพใน message (`build_messages(img)`) แทน placeholder ทั้งใน `full` และ `prompt_only`

- [ ] **Step 4: เทรนเต็ม**

Run: `HF_TOKEN=$(cat ~/.cache/huggingface/token) colab/run_job.sh colab/jobs/train.sh L4 300`
Expected: `EXIT_CODE=0`, `uploaded → <HF_USER>/dvl-qwen35-2b-lora`, eval_loss ต่ำสุดก่อน early stop · จดเวลาเทรน + CU ที่ใช้ (`colab usage` ก่อน/หลัง)

- [ ] **Step 5: Predict ด้วย adapter แล้วเทียบกับ base**

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token) ADAPTER=<HF_USER>/dvl-qwen35-2b-lora TAG=lora_v1 \
  colab/run_job.sh colab/jobs/predict.sh L4 60
../iron-coach-th/.venv/bin/python scripts/evaluate.py data/dataset_v1/test_gold.jsonl \
  runs/dvl-predict-*/out/predictions-base.jsonl runs/dvl-predict-*/out/predictions-lora_v1.jsonl
```
Expected: ตารางเทียบ 2 คอลัมน์ · **เกณฑ์ผ่าน:** `type_macro_f1` ของ lora_v1 > base, `false_alarm_rate` และ `miss_rate` ไม่แย่กว่า base, `json_valid_rate` ≥ 0.99 — ถ้าไม่ผ่าน **ห้ามสรุปว่าดีขึ้นจาก loss** ให้ดู per-type F1 ว่าคลาสไหนพัง แล้วกลับไปปรับ balance (บทเรียน exp05)

- [ ] **Step 6: เขียน `README.md`** — ภาษาไทย: เป้าหมาย, ชุดประเภทเฟส 1, วิธีรัน (คำสั่ง `run_job.sh` แต่ละขั้น), ตาราง base vs lora_v1, per-type F1, อัตราที่ teacher label ผิด (จาก Task 10), CU ที่ใช้, ข้อจำกัด (ไม่มีภาพไทย, คลาสที่ test < 30 ภาพ, ไลเซนส์ NC)

- [ ] **Step 7: Commit**

```bash
git add scripts/train.py colab/jobs/train.sh reports/lora_v1.json README.md
git commit -m "feat: LoRA fine-tune Qwen3.5-2B + base vs tuned eval"
```

---

## นอกขอบเขตแผนนี้ (แผนถัดไป)

- **ใช้งานบนเครื่อง 6GB:** merge adapter → GGUF + mmproj ด้วย `llama.cpp/convert_hf_to_gguf.py` (Qwen3.5 มี `tie_word_embeddings=True` — ระวังแบบเดียวกับ iron-coach) → `llama-server` (มี binary Vulkan อยู่ที่ `olmocr-lab/bin/`) + `serve.py` ที่คืน `to_api()` และคำนวณ confidence จาก `logprobs`
- **ขยายกลุ่ม B** (`fallen_tree`, `fallen_power_line`, `motorcycle_accident`, `dangerous_animal` …) ด้วย teacher ติด label บนภาพเฮอริเคน/อุบัติเหตุเดิม + เก็บภาพเพิ่ม
- **ชุดทดสอบภาพจากไทย** 100–300 ภาพ (น้ำท่วมภาคเหนือ ไฟป่าดอย ฯลฯ) — ตัววัดที่สำคัญที่สุดก่อนใช้จริง
- เชื่อมเข้า SOS backend (ใช้ `incident_type` เป็นคำแนะนำ/ตรวจทานประเภทที่ผู้แจ้งเลือก ไม่ใช่ตัดสินแทน)
