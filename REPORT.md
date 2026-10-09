# รายงานโมเดล: dvl-qwen3.5-2b (ภาพแจ้งเหตุ → JSON ตาม SOS catalog)

รายงานนี้เป็น model report / model card ของ **LoRA v1 (`lora_v1`)** ที่ fine-tune จาก `Qwen/Qwen3.5-2B`
และเวอร์ชัน GGUF ที่ใช้รันจริงบนเครื่อง เขียนให้คนสองกลุ่มอ่าน: ทีมที่กำลังตัดสินใจว่าจะเอาโมเดลไปต่อกับระบบแจ้งเหตุ SOS หรือไม่
และคนที่อยากรู้ว่าโมเดลนี้สร้างมาอย่างไร ตัวเลขทุกตัวมาจากไฟล์ใน `reports/` และผลรันที่บันทึกไว้แล้ว ไม่ได้รันใหม่ตอนเขียนรายงาน
(path ในรายงานนับจาก root ของ repo)

---

## 1. สรุป (TL;DR)

- **ทำอะไร:** รับภาพ 1 ภาพ แล้วตอบ JSON บรรทัดเดียว `{"category", "incident_type", "severity"}` ตามคลังประเภทเหตุของ SOS
  (12 ประเภทที่ดูจากภาพได้ + `unsure` + `no_incident`) และให้ `confidence` ที่คำนวณจาก logprob ของ token
- **ดีขึ้นจาก base ชัดเจน:** macro-F1 บน gold (247 ภาพที่คนตรวจ) 0.525 → 0.670 (bf16) / 0.689 (GGUF Q4_K_M),
  บน test เต็ม (3,244 ภาพ) 0.321 → 0.607 · false alarm ลดจาก 0.400 → 0.080–0.160 (gold) และ 0.484 → 0.117 (test)
- **แต่ตัวโมเดลเปล่า ๆ พลาดเหตุจริงมากกว่า base** (miss: gold 0.018 → 0.036, test 0.007 → 0.025) จึงต้องใช้คู่กับ
  **abstain gate**: ถ้าโมเดลตอบ `no_incident` แต่ P(เป็นเหตุ) ≥ T = 3.2e-5 ให้เปลี่ยนเป็น `unsure` แล้วส่งให้คนดู
  ผลบน Q4_K_M: miss บน gold 0.045 → 0.018 (เท่า base) และบน test ที่ไม่รวม gold 0.031 → 0.003 โดยต้องส่งให้คนดูราว 5–10% ของภาพ
- **ตัวเลขหลัก (GGUF Q4_K_M + gate, ตัวที่แนะนำให้ใช้):**

| ชุด | macro-F1 | false_alarm | miss | ส่งให้คนดู |
|---|---|---|---|---|
| gold 247 (คนตรวจ) | 0.697 | 0.080 | 0.018 | 5.3% |
| test ไม่รวม gold 2,997 | 0.715 (8 คลาส*) | 0.130 | 0.003 | 10.4% |

  \* test ที่ตัด gold ออกเหลือแค่ 8 คลาส (คลาสหายากถูกดึงเข้า gold ทั้งหมด) จึงเทียบ macro-F1 กับตารางอื่นตรง ๆ ไม่ได้ — ดูหัวข้อ 5.5

- **คำแนะนำ:** ใช้ **GGUF Q4_K_M + abstain gate** (ผ่าน HTTP API ใน `api/`) ในฐานะ **ผู้ช่วยคัดกรองที่มีคนตรวจซ้ำ**
  ใช้เติมประเภทเหตุให้อัตโนมัติและจัดลำดับคิว **ไม่ใช่** ตัวตัดสินสั่งการเอง ห้ามทิ้งภาพที่โมเดลบอกว่า "ไม่ใช่เหตุ" โดยไม่มีคนดู
  และยังไม่เคยทดสอบกับภาพจากประเทศไทยเลย ก่อนใช้จริงต้องทดสอบกับภาพแจ้งเหตุจริงของพื้นที่และทบทวนไลเซนส์ข้อมูล (หัวข้อ 7)

---

## 2. งานและขอบเขต

**Input:** ภาพ 1 ภาพ (jpeg/png/webp) ย่อให้ด้านยาวสุด ≤ 512 px แบบเดียวกับตอนสร้าง dataset (`dvl/imgutil.py`)
คู่กับ system prompt และคำสั่งที่อยู่ที่เดียวใน `dvl/prompt.py`

**Output ของโมเดล:** JSON compact เรียง key ตายตัว เช่น `{"category":"disaster","incident_type":"flood","severity":"mild"}`

- `severity` คือสิ่งที่เห็นในภาพ: `none` (เฉพาะ `no_incident`) / `mild` / `severe` — **ไม่ใช่** ระดับ 2–4 ของ catalog
- `confidence` ไม่ได้ให้โมเดลเขียนเอง แต่ server คำนวณจากผลคูณความน่าจะเป็นของ token ค่า `incident_type` (`dvl/confidence.py`)
- ถ้าคำตอบพัง (ไม่ใช่ JSON, ค่านอกชุด) parser จะไม่ crash แต่คืน `unsure` + `valid=false` (`dvl/schema.py`)

**ชุดประเภท (`dvl/catalog.py`)** — key ตรงกับ SOS incident catalog:

| category | incident_type |
|---|---|
| `fire_hazard` | `forest_fire`, `building_fire`, `vehicle_fire`, `smoke_detected` |
| `disaster` | `flood`, `storm`, `landslide`, `damaged_structure`, `road_hazard` |
| `rescue` | `building_collapse`, `water_rescue` |
| `accident` | `vehicle_collision` |
| `other` | `unsure` (มีใน catalog จริง) |
| `null` | `no_incident` (ค่าของโมเดลเท่านั้น ไม่มีใน SOS = ไม่ต้องสร้างเหตุ) |

**นอกขอบเขต:**

- ประเภทเหตุอื่นใน SOS catalog ที่ดูจากภาพไม่ได้หรือไม่มีข้อมูล (เช่น เหตุทางการแพทย์ อาชญากรรม สัตว์) — โมเดลควรตอบ `unsure`
  แต่**ยังไม่เคยวัด**ว่ามันตอบ `unsure` จริงกับภาพแบบนั้น
- `earthquake`: ภาพบอกได้แค่ว่าอาคารเสียหายหรือถล่ม ไม่ได้บอกสาเหตุ จึง label เป็น `damaged_structure` / `building_collapse`
- การระบุตำแหน่ง จำนวนผู้บาดเจ็บ ความเร่งด่วนเชิงปฏิบัติการ หรือการอ่านข้อความในภาพ
- วิดีโอ หรือหลายภาพต่อเหตุการณ์

---

## 3. ข้อมูล

### 3.1 แหล่งข้อมูล

รวมภาพจาก 7 dataset บน Hugging Face ได้ candidate pool 27,396 ภาพ (`scripts/build_candidates.py`) แล้วใช้จริงในชุด train ตามตาราง
ไลเซนส์คัดมาจาก model card ที่ `scripts/export_gguf.py` สร้าง:

| Dataset | ใช้ทำอะไร | วิธี label | pool | ใน train | ไลเซนส์ |
|---|---|---|---|---|---|
| `QCRI/CrisisMMD` | ภาพเหตุจริงจาก tweet 7 เหตุการณ์ | event → candidates → teacher เลือก | 18,082 | 3,734 | **CC BY-NC-SA 4.0** |
| `anwan/DisasterVQA` | เติม fire/landslide/accident (ภาพจาก MEDIC + Incidents1M) | disaster_type → candidates → teacher | 1,395 | 996 | CC BY-SA 4.0 (ภาพต้นทางมีเงื่อนไขของตัวเอง) |
| `AbdullahImran/balanced_wildfire_dataset` | `forest_fire` + severity | ตายตัว | 4,882 | 1,368 | other |
| `fireviewer/fire_and_smoke_detection_very_hard_negative` | hard negative (พระอาทิตย์ตก หมอก ไฟถนน) | `no_incident` ตายตัว | 195 | 17 | other |
| `hiennguyen9874/traffic-accident-detection` | `vehicle_collision` มุม CCTV | ตายตัว (cap 800/คลาส) | 1,456 | 733 | ไม่ระบุ |
| `Arpitraj01/Pothole_classification` | `road_hazard` + severity | ตายตัว | 386 | 289 | MIT |
| `nlphuji/flickr_1k_test_image_text_retrieval` | ภาพชีวิตประจำวัน กัน false alarm | `no_incident` ตายตัว | 1,000 | 76 | ไม่ระบุ |

ที่มา: `scripts/build_candidates.py`, `scripts/make_splits.py`, `scripts/export_gguf.py` (การ์ดไลเซนส์) · CrisisMMD เป็น 66% ของ pool

### 3.2 การติด label ด้วย teacher

- แต่ละ dataset ถูก map เป็น "ชุดผู้สมัคร" (candidates) ของ incident_type (`dvl/mapping.py`) ถ้ามีผู้สมัครตัวเดียวถือเป็น label ตายตัว (`label_source = "fixed"`)
- ถ้ามีหลายตัว ให้ teacher **`Qwen/Qwen3.5-9B`** เลือกจากผู้สมัคร (+ `unsure`/`no_incident` เสมอ) → `label_source = "teacher"`
  แถวที่ teacher ตอบนอกชุดผู้สมัคร (202) หรือมั่นใจ < 0.5 (32) ถูกตัดทิ้ง
- งบ compute จำกัด teacher ไว้ 9,500 calls (~5.8 ชม. บน L4, 2.20 s/ภาพ): แถว CrisisMMD ที่มีหลายผู้สมัครถูกสุ่มเหลือ 5,594 จาก 11,121 (seed 20261007)
  แถวอื่นเก็บครบ ได้ภาพที่มี label ทั้งหมด 21,635 ภาพ
- ข้อสังเกต: teacher ถูกจำกัดให้เลือกเฉพาะจากผู้สมัคร label จึงผิดได้ทั้งจาก teacher และจาก mapping

### 3.3 การแบ่งชุด (`scripts/make_splits.py`)

- รวมภาพเกือบซ้ำ (aHash 64-bit, Hamming ≤ 4 แบบเทียบทุกคู่) ไว้กลุ่มเดียวกัน แล้วแบ่ง stratified 80/5/15 ทั้งกลุ่ม → ภาพรีทวีตซ้ำไม่รั่วข้าม split
- balance **เฉพาะ train**: cap 1,200 ภาพต่อคลาส และ `no_incident` ≤ 25% ของ train · val/test คงสัดส่วนธรรมชาติ

| split | n | `no_incident` | fixed / teacher |
|---|---|---|---|
| train | 7,213 | 16.6% (1,200) | 3,123 / 4,090 |
| val | 1,082 | 73.6% (796) | 764 / 318 |
| test | 3,244 | 74.3% (2,409) | 2,297 / 947 |

จำนวนต่อคลาสใน **train**: damaged_structure 1200, forest_fire 1200, no_incident 1200, storm 1109, flood 807, vehicle_collision 726,
road_hazard 435, building_collapse 172, building_fire 139, water_rescue 111, landslide 41, smoke_detected 34, unsure 20, vehicle_fire 19

จำนวนต่อคลาสใน **test**: no_incident 2409, damaged_structure 168, forest_fire 163, storm 143, flood 118, vehicle_collision 98,
road_hazard 69, building_collapse 29, water_rescue 18, building_fire 16, landslide 5, smoke_detected 4, vehicle_fire 2, unsure 2

### 3.4 Gold set ที่คนตรวจ (247 ภาพ)

- สุ่มจาก test สูงสุด 25 ภาพต่อคลาส (คลาสที่มีน้อยกว่านั้นได้ทุกภาพ) → 247 ภาพ: teacher 176, fixed 71 (`scripts/review_sheet.py`)
- ผู้ใช้ตรวจทั้ง 247 ภาพเมื่อ 2026-10-08 แล้ว **ยอมรับทุก label โดยไม่แก้เลย** (0/176 teacher, 0/71 fixed)
  นี่คือ "การยอมรับ" ไม่ใช่หลักฐานว่า label ถูกทั้งหมด: ไม่มีผู้ตรวจคนที่สอง ไม่ได้วัด inter-annotator agreement
  และผู้ตรวจเห็น label เดิมก่อนตัดสิน ซึ่งเอนไปทางยอมรับได้
- gold เป็นชุดที่**สมดุลคลาส** ส่วน test เป็นชุดเบ้ (74% `no_incident`) ค่า macro-F1 ของสองชุดจึงเทียบข้ามกันไม่ได้

---

## 4. การเทรน

| รายการ | ค่า |
|---|---|
| base model | `Qwen/Qwen3.5-2B` (Apache-2.0), bf16, ไม่เปิด thinking |
| วิธี | LoRA บน bf16 (ไม่ใช่ QLoRA) · `scripts/train.py` ด้วย `trl` SFTTrainer บน Colab **L4** |
| LoRA | r=16, alpha=32, dropout 0.05 · ไม่แตะ vision encoder |
| target modules | full-attention: `q_proj`/`k_proj`/`v_proj`/`o_proj` (6 ชั้น) · linear-attention: `in_proj_qkv`/`in_proj_z`/`out_proj` (18 ชั้น) · MLP: `gate_proj`/`up_proj`/`down_proj` (24 ชั้น) |
| trainable | 15,630,336 / 2,228,872,000 = **0.70%** |
| loss | เฉพาะ token คำตอบ JSON ของ assistant (`{json}<\|im_end\|>`) — prompt และภาพไม่นับ loss |
| optimizer | LR 1e-4 cosine, warmup 3%, batch 4 × grad accum 4 (16 ภาพ/step), gradient checkpointing |
| รอบ | 2 epoch = 902 step · eval/save ทุก 100 step · early stopping patience 3 (ไม่ทำงาน) |
| เวลา | train_runtime 12,266 s ≈ **3.4 ชม.** บน L4 · train_loss เฉลี่ย 0.047 |
| เลือก checkpoint | eval_loss ต่ำสุดบน val → **step 700** (eval_loss 0.0239) |

Qwen3.5 มีแค่ 6 ชั้นที่เป็น full attention ส่วนอีก 18 ชั้นเป็น linear attention ซึ่งใช้ชื่อ layer ต่างออกไป
ถ้าใส่แค่ `q/k/v/o_proj` ตามสูตรปกติ LoRA จะไม่แตะ attention ของชั้นส่วนใหญ่ จึงต้องเพิ่ม `in_proj_qkv|in_proj_z|out_proj` เข้า regex
(`in_proj_a/b` ข้ามเพราะเล็กมาก)

**eval_loss บน val** (จาก `log_history.json` ของรอบเทรน ไม่ได้ commit):

| step | 100 | 200 | 300 | 400 | 500 | 600 | **700** | 800 | 900 | 902 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval_loss | 0.0375 | 0.0633 | 0.0301 | 0.0398 | 0.0294 | 0.0299 | **0.0239** | 0.0277 | 0.0271 | 0.0271 |

ข้อควรระวัง: val มี `no_incident` 74% ขณะที่ train มีแค่ 17% eval_loss จึงให้รางวัลกับการตอบ `no_incident` เป็นหลัก
และกราฟก็ไม่ลดลงอย่างสม่ำเสมอ (กระโดดที่ step 200) checkpoint ที่ eval_loss ต่ำสุดจึงไม่จำเป็นต้องเป็นตัวที่ miss ต่ำสุด
ซึ่งน่าจะเป็นส่วนหนึ่งที่ทำให้ miss เพิ่มขึ้น (หัวข้อ 5.1)

**Compute (Colab compute units, L4 ≈ 1.54 CU/ชม.):** เทรนรวม smoke ≈ 5.8 CU, predict lora_v1 บน test ≈ 2.7 CU (รวมรอบที่ connection หลุด ~0.4 CU),
export GGUF บน CPU < 0.5 CU · ยอดคงเหลือของบัญชีลดจาก 185.62 → 176.98 CU ช่วงเทรนถึง export (8.64 CU)
ก่อนหน้านั้นมีงานสร้าง dataset, teacher (~5.8 ชม. L4) และ baseline predict (~109 นาที L4) ซึ่งรวมแล้วราว 14 CU ถ้านับจากโควตาเริ่มต้น 200 CU
**รวมทั้งโปรเจกต์ประมาณ 23 CU** (ค่านี้ประมาณจากยอดคงเหลือของบัญชี ไม่ได้วัดรายงาน)

---

## 5. ผลการประเมิน

นิยามตัวชี้วัด (`dvl/metrics.py`):

- **macro-F1** เฉลี่ย F1 ของ `incident_type` ทุกคลาสที่**มีอยู่ใน label ของชุดนั้น** และเป็นตัวชี้วัดหลัก
- **false_alarm** = สัดส่วนของภาพ `no_incident` ที่โมเดลตอบว่าเป็นเหตุ
- **miss** = สัดส่วนของภาพเหตุจริงที่โมเดลตอบว่า `no_incident` ซึ่งเป็นความผิดพลาดที่อันตรายที่สุดในงานนี้
- **ECE** วัด calibration ของ `confidence` (10 bins)
- generation ทุกรอบเป็น greedy, `max_new_tokens 64`, ไม่เปิด thinking

### 5.1 base vs lora_v1 (bf16, HF transformers บน L4)

| metric | gold base | gold lora_v1 | test base | test lora_v1 |
|---|---|---|---|---|
| n | 247 | 247 | 3,244 | 3,244 |
| json_valid_rate | 1.000 | 1.000 | 0.997 | 1.000 |
| type_accuracy | 0.656 | 0.794 | 0.551 | 0.879 |
| **type_macro_f1** | 0.525 | **0.670** | 0.321 | **0.607** |
| category_accuracy | 0.765 | 0.858 | 0.580 | 0.893 |
| severity_accuracy | 0.794 | 0.858 | 0.588 | 0.871 |
| **false_alarm** | 0.400 | **0.160** | 0.484 | **0.117** |
| **miss** | 0.018 (4/222) | **0.036 (8/222)** | 0.007 (6/835) | **0.025 (21/835)** |
| ECE | 0.246 | 0.173 | 0.324 | 0.105 |

ที่มา: `reports/base-gold.json`, `reports/lora_v1-gold.json`, `reports/base-test.json`, `reports/lora_v1-test.json`

เกณฑ์ที่ตั้งไว้ก่อนเทรน: macro-F1, false alarm และ JSON ผ่านทั้งสองชุด แต่ **miss ไม่ผ่าน** (สูงกว่า base)
miss ของ base ต่ำเพราะ base แทบไม่ตอบ `no_incident` เลย (false alarm 0.48) ส่วน lora_v1 กล้าตอบว่าไม่ใช่เหตุมากขึ้น
ภาพเหตุที่ lora_v1 พลาดเป็น `no_incident` บน test 21 ภาพ ได้แก่ storm 6, road_hazard 5, damaged_structure 3, flood 3, forest_fire 2 และ unsure 2

**Per-type F1 บน test (base → lora_v1, support)** — `reports/base-test.json`, `reports/lora_v1-test.json`:

| type | n | F1 | | type | n | F1 |
|---|---|---|---|---|---|---|
| no_incident | 2409 | 0.68 → 0.93 | | building_collapse | 29 | 0.09 → 0.58 |
| damaged_structure | 168 | 0.05 → 0.73 | | water_rescue | 18 | 0.10 → 0.59 |
| forest_fire | 163 | 0.64 → 0.90 | | building_fire | 16 | 0.56 → 0.63 |
| storm | 143 | 0.52 → 0.66 | | landslide | 5 | 0.09 → 0.46 |
| flood | 118 | 0.51 → 0.76 | | smoke_detected | 4 | 0.00 → 0.00 |
| vehicle_collision | 98 | 0.68 → 0.91 | | vehicle_fire | 2 | 0.25 → 0.67 |
| road_hazard | 69 | 0.33 → 0.68 | | unsure | 2 | 0.00 → 0.00 |

ทุกคลาสดีขึ้นหรือเท่าเดิม ยกเว้น `smoke_detected` และ `unsure` ที่ยังเป็น 0 (train มีแค่ 34 และ 20 ภาพ)
`storm` precision ต่ำ (0.53) คือโมเดลชอบตอบ storm กับภาพที่ไม่ใช่ storm ส่วนบน gold มีสองคลาสที่ลดลงเล็กน้อย:
building_fire 0.86 → 0.79 (n=16) และ building_collapse 0.62 → 0.61 (n=25)

**แยกตาม label_source บน test** (macro-F1 ในกลุ่มย่อยเฉลี่ยเฉพาะคลาสที่มีในกลุ่มนั้น จึงสูงกว่าค่ารวมได้):

| group | n | macro-F1 | false_alarm | miss |
|---|---|---|---|---|
| fixed | 2,297 | 0.450 → 0.704 | 0.498 → 0.126 | 0.004 → 0.012 |
| teacher | 947 | 0.469 → 0.676 | 0.411 → 0.068 | 0.009 → 0.031 |

แถวที่ teacher ติด label คือแถวที่ teacher มั่นใจ (≥ 0.5) จึงอาจง่ายกว่าภาพยากจริง และโมเดลที่เรียนจาก teacher ก็อาจได้คะแนนดีบนแถวพวกนี้เพราะเลียนแบบความผิดของ teacher ได้

### 5.2 ผลของ quantization (GGUF ผ่าน llama.cpp, gold 247)

| metric | lora_v1 bf16 (HF) | GGUF Q8_0 | GGUF Q4_K_M |
|---|---|---|---|
| type_macro_f1 | 0.670 | 0.690 | 0.689 |
| type_accuracy | 0.794 | 0.798 | 0.798 |
| severity_accuracy | 0.858 | 0.874 | 0.879 |
| false_alarm | 0.160 | 0.120 | 0.080 |
| miss | 0.036 | 0.041 | 0.045 |
| json_valid_rate | 1.000 | 1.000 | 1.000 |
| ECE | 0.173 | 0.160 | 0.158 |

ที่มา: `reports/lora_v1-gold.json`, `reports/gguf_q8-gold.json`, `reports/gguf_q4-gold.json`

บนชุดเล็กแบบนี้ **ไม่เห็นว่า quantization ทำให้แย่ลง** ความต่างระหว่าง Q8 กับ Q4 เป็นแค่ 1–2 ภาพต่อคลาส
(false alarm 0.12 กับ 0.08 ต่างกันแค่ 1 ภาพจาก 25) ส่วนที่ GGUF ดูดีกว่า bf16 นิดหน่อยไม่ควรอ่านว่า "quantize แล้วดีขึ้น"
เพราะ runtime ต่างกัน (HF processor กับ llama.cpp + mmproj ใช้ตัวถอดและย่อภาพคนละตัว) และอยู่ในระดับ noise ของ n=247
ข้อจำกัดคือยังไม่ได้วัด GGUF บน test เต็ม 3,244 ภาพในแบบไม่มี gate

### 5.3 Abstain gate (GGUF Q4_K_M)

**กติกา:** ถ้าโมเดลตอบ `no_incident` แต่ `p_incident = 1 − P(category = null) ≥ T` ให้เปลี่ยนคำตอบเป็น `unsure` / `other` / `mild` แล้วตั้ง `needs_review`
เกณฑ์เลือก T: ใช้ค่า T ที่ใหญ่ที่สุดที่ทำให้ miss บน **val** ≤ 1% ได้ **T = 3.2e-5** จากนั้นวัดบน gold และ test ที่ตัด gold ออก ซึ่งไม่ได้ใช้เลือก T
(`scripts/abstain_sweep.py`, ผลเต็มใน `reports/abstain-q4.md`)

| ชุด | gate | macro-F1 | false_alarm | miss | ส่งให้คนดู |
|---|---|---|---|---|---|
| val 1,082 (ใช้เลือก T) | ปิด → เปิด | 0.674 → 0.671 | 0.142 → 0.142 | 0.024 → **0.003** | 0% → 9.7% |
| gold 247 | ปิด → เปิด | 0.684 → 0.697 | 0.080 → 0.080 | 0.045 → **0.018** | 0% → 5.3% |
| test ไม่รวม gold 2,997 | ปิด → เปิด | 0.724 → 0.715 | 0.130 → 0.130 | 0.031 → **0.003** | 0.1% → 10.4% |

- gate ไม่เปลี่ยน false alarm เพราะมันแตะเฉพาะภาพที่โมเดลตอบ `no_incident`
- miss บน gold กลับมาเท่า base (0.018) และบน test ต่ำกว่า base (0.003 เทียบ 0.007) ส่วน false alarm ยังต่ำกว่า base 3–5 เท่า
- **ต้นทุน:** บน test ที่ตัด gold ออกมีภาพ `unsure` 311 ภาพ (gate เปลี่ยน 308 + โมเดลตอบ `unsure` เอง 3)
  ในนี้เป็นเหตุจริง 17 ภาพ (storm 7, road_hazard 4, damaged_structure 2, forest_fire 2, flood 2) และไม่ใช่เหตุ 294 ภาพ
  บน gold gate ส่งให้คนดู 13 ภาพ เป็นเหตุจริง 6 ภาพ คือราว 1 ใน 18 ของภาพที่ส่งให้คนดูบน test เป็นเหตุจริง ซึ่งเป็นราคาของการลด miss
- ค่า T เล็กมาก (ระดับ 1e-5) เพราะโมเดล**มั่นใจเกินจริง** เมื่อตอบ `no_incident` ก็มักให้ P(null) ใกล้ 1 มาก
  จึงต้องหาสัญญาณความลังเลจากส่วนที่เหลือเล็ก ๆ
- logprob ได้มาจาก top-5 เท่านั้น ถ้า `null` หลุด top-5 จะถือว่าเป็นเหตุ ซึ่งพลาดไปทางปลอดภัย (ส่งให้คนดู)

### 5.4 ผลผ่าน HTTP API จริง (gold 247, RTX 2060 6GB)

API (`api/app.py`) ห่อ llama-server Q4_K_M + mmproj และใช้ gate ทุกคำขอ ภาพถูก `to_rgb_resized(512)` แล้วส่งต่อเป็น PNG (lossless)

| | API (PNG) | API รอบแรก (JPEG q95) | สคริปต์ `predict_gguf.py` + gate |
|---|---|---|---|
| `incident_type` ตรงกับสคริปต์ | 240/247 (97.2%) | 234/247 (94.7%) | – |
| ตอบถูกตาม gold | 196 | 192 | 191 |
| miss | 4/222 | 3/222 | 4/222 |
| false alarm | 2/25 | 2/25 | 2/25 |
| ส่งให้คนดู | 11 | 14 | 13 |
| latency p50 / p90 / max (ms) | 503 / 553 / 926 | 444 / 490 / 792 | – |

ที่มา: README หัวข้อ "HTTP API" (ส่วนผลตรวจจริงบนเครื่อง) · รอบ JPEG ไม่ได้ใช้แล้ว (ดูบทเรียนข้อ 8.4)

**ความเร็วและหน่วยความจำ (RTX 2060 6GB, llama.cpp b10909 Vulkan, VRAM วัดทั้งการ์ดรวม desktop ~0.3 GB):**

| | s/ภาพ | VRAM peak (MiB) |
|---|---|---|
| base bf16 (HF บน Colab L4) | 2.01 | – |
| lora_v1 bf16 (HF บน Colab L4) | 1.70 | – |
| GGUF Q8_0 (`predict_gguf.py`) | 0.58 | 3,294 |
| GGUF Q4_K_M (`predict_gguf.py`) | 0.46 | 2,543 (2,828 ในรอบ test เต็ม) |
| API Q4_K_M (PNG, ต่อคำขอ) | ~0.50 (p50) | 2,579 |

API ประมวลผลทีละคำขอ (`-np 1`) ถ้ามีหลายคำขอเข้ามาพร้อมกันจะต่อคิว throughput จึงราว 2 ภาพ/วินาทีบนการ์ดนี้

### 5.5 อ่านตัวเลขอย่างไรให้ไม่หลง

- **อย่าเทียบ macro-F1 ข้ามชุด** เพราะ gold สมดุล ส่วน test เบ้ ส่วน "test ไม่รวม gold" ไม่มีคลาสหายากเหลือเลย
  (gold ดึงภาพของ landslide, smoke_detected, unsure, vehicle_fire, water_rescue และ building_fire ไปหมดแล้ว)
  macro-F1 0.715–0.724 ของชุดนั้นจึงเฉลี่ยแค่ 8 คลาส และดูสูงกว่าตารางอื่นเพราะคลาสยากหายไป ไม่ใช่เพราะโมเดลเก่งขึ้น
- **accuracy บน test ดูดีเกินจริง** (0.879) เพราะ 74% ของ test เป็น `no_incident`
- **คลาสหายากมี 2–5 ภาพ** ภาพเดียวก็เปลี่ยน F1 ได้ 0.2–0.5
- **ผลข้ามรอบของ llama-server ไม่ deterministic เป๊ะ** Q4 บน gold แบบไม่มี gate ได้ macro-F1 0.689 ใน `reports/gguf_q4-gold.json`
  แต่ได้ 0.684 ในรอบที่เก็บ top-5 logprobs สำหรับ gate (`reports/abstain-q4.md`) ทั้งที่เป็นไฟล์โมเดลเดียวกัน

---

## 6. การใช้งาน (deploy)

| ชิ้นส่วน | ที่อยู่ | หมายเหตุ |
|---|---|---|
| LoRA adapter (checkpoint-700) | https://huggingface.co/Petanque/dvl-qwen35-2b-lora | **private** ขอสิทธิ์เข้าถึงได้ |
| GGUF (merged) | https://huggingface.co/Petanque/dvl-qwen3.5-2b-gguf | **private** ขอสิทธิ์เข้าถึงได้ · `dvl-qwen3.5-2b-Q4_K_M.gguf` 1.19 GiB, `dvl-qwen3.5-2b-Q8_0.gguf` 1.87 GiB, `mmproj-dvl-qwen3.5-2b-F16.gguf` 0.62 GiB (จำเป็นสำหรับภาพ) |
| inference server | llama.cpp `llama-server` **b10909** (`a2878d30d`) | `--jinja --reasoning off -c 4096 -np 1` ดูคำสั่งเต็มใน README หัวข้อ "Run locally (GGUF)" |
| HTTP API | `api/app.py`, `api/run.sh` | `POST /v1/classify`, `GET /v1/labels`, `GET /health` · คำตอบมี `needs_review`, `p_incident`, `model_incident_type` · ดู README หัวข้อ "HTTP API" |
| Docker | ดู README หัวข้อ Docker | |
| Bruno collection | `bruno/` | contract ของทุก endpoint พร้อมตัวอย่าง request/response และชุดเทสที่รันได้ |

แนวทางเชื่อมกับระบบ SOS:

1. เรียก `/v1/classify` แล้วใช้ `incident_type` / `category` ที่ได้มา**เติมฟอร์มล่วงหน้า**ให้เจ้าหน้าที่ยืนยัน
2. `needs_review = true` หมายความว่า "อาจเป็นเหตุ ให้คนดู" **ห้ามตีความว่าไม่ใช่เหตุ** และห้ามทิ้ง
3. แม้ได้ `no_incident` และ `needs_review = false` ก็ยังมี miss ราว 0.3–2% (หัวข้อ 5.3) ดังนั้นห้ามปิดเรื่องแจ้งเหตุจากผลนี้อย่างเดียว
4. **อย่าใช้ `confidence` แทน `needs_review`** เพราะ `confidence` ของ `incident_type` มักอิ่มที่ ~1.0 และ calibration ยังไม่ดี (ECE 0.16)
5. ค่า T = 3.2e-5 **ผูกกับไฟล์ Q4_K_M + llama.cpp b10909** ถ้าเปลี่ยน quant, เวอร์ชัน llama.cpp หรือ preprocessing ต้อง sweep T บน val ใหม่
6. ค่าเริ่มต้น bind ที่ 127.0.0.1 ถ้าเปิดให้เครื่องอื่นเรียกต้องตั้ง `DVL_API_KEY` ด้วย

---

## 7. ข้อจำกัดและความเสี่ยง

- **ไม่มีภาพจากประเทศไทย:** ภาพทั้งหมดมาจาก dataset ต่างประเทศ (CrisisMMD เป็นภาพจาก tweet ของเฮอริเคนในสหรัฐฯ/แคริบเบียน ไฟป่าแคลิฟอร์เนีย แผ่นดินไหวเม็กซิโกและอิรัก-อิหร่าน และน้ำท่วมศรีลังกา)
  ภาพแจ้งเหตุจริงในไทยต่างกันทั้งสภาพแวดล้อม ประเภทอาคาร รถ และสไตล์การถ่ายด้วยมือถือ จึง**ยังไม่รู้ผลบนภาพไทย** ต้องเก็บชุดทดสอบไทยก่อนใช้งานจริง
- **ไลเซนส์ NC:** CrisisMMD เป็น CC BY-NC-SA 4.0 และเป็นส่วนใหญ่ของ train บางชุดไม่ระบุไลเซนส์ และภาพของ DisasterVQA มาจาก MEDIC/Incidents1M ซึ่งมีเงื่อนไขของตัวเอง
  น้ำหนักโมเดลจึงควรถือว่าใช้ได้แบบ **non-commercial / share-alike** เท่านั้น
  ถ้าจะใช้ในระบบจริงของหน่วยงานท้องถิ่น (อบต./เทศบาล) **ต้องทบทวนไลเซนส์ใหม่ทั้งหมด** หรือเทรนใหม่ด้วยข้อมูลที่ใช้ได้
  ด้วยเหตุนี้ repo นี้จึงไม่แจกจ่ายภาพจาก dataset (รวมถึงภาพตัวอย่างใน `bruno/fixtures/`)
- **คลาสหายาก:** test มี smoke_detected 4, vehicle_fire 2, unsure 2, landslide 5 ภาพ ซึ่ง F1 ของ `smoke_detected` และ `unsure` เป็น **0**
  (bf16 บน test; Q4 บน gold ได้ smoke_detected 0.20 จากภาพที่ถูกแค่ 1 ภาพ) จึงไม่ควรเชื่อผลของคลาสพวกนี้เลย
  `unsure` ในความหมาย "เหตุที่อยู่นอก 12 ประเภท" ก็แทบไม่ได้ถูกสอน
- **มั่นใจเกินจริง:** threshold ของ gate อยู่ที่ระดับ 1e-5 และ ECE 0.10–0.17 แปลว่าความน่าจะเป็นดิบของโมเดลไม่ควรนำไปอ่านเป็นความน่าจะเป็นจริง
- **T เปราะ:** ผูกกับ quant, เวอร์ชัน llama.cpp และวิธีส่งภาพ แค่ JPEG encode ซ้ำรอบเดียวก็เปลี่ยนคำตอบไป ~5% และ server ไม่ deterministic เป๊ะข้ามรอบ
- **label noise จาก teacher:** label 29% ของ test และ 57% ของ train มาจาก Qwen3.5-9B การตรวจ gold ที่ "ยอมรับทั้งหมด" ไม่ได้พิสูจน์ว่า teacher ถูก
  label ตายตัวจาก mapping ก็ผิดได้ (เช่น ภาพจาก tweet ของเหตุไฟป่าที่จริง ๆ เป็นภาพแผนที่)
- **miss กับ false alarm แลกกัน:** gate ลด miss ด้วยการส่งภาพให้คนดูเพิ่มขึ้น ~10% ถ้าลดภาระคน miss ก็จะกลับมา
  ระดับที่เหมาะสมขึ้นกับว่าทีมรับภาระตรวจได้แค่ไหน ซึ่งต้องตัดสินใจร่วมกับผู้ใช้จริง
- **severity** วัดเฉพาะสิ่งที่เห็นในภาพแบบ 3 ระดับ (accuracy ~0.86–0.88) ไม่ได้บอกความเร่งด่วนเชิงปฏิบัติการ
- **ภาพที่ตั้งใจหลอก:** ยังไม่เคยทดสอบ adversarial หรือภาพที่สร้างด้วย AI รวมถึงภาพหน้าจอหรือภาพข่าวเก่าที่ส่งซ้ำ
- **ไม่ใช่ระบบสั่งการอัตโนมัติ:** ไม่ควรใช้ผลของโมเดลอย่างเดียวในการตัดสินว่าจะส่ง หรือจะไม่ส่ง เจ้าหน้าที่ออกไป

---

## 8. บทเรียนระหว่างทาง

1. **confidence ของ `incident_type` อิ่มที่ 1.0** เพราะ JSON ขึ้นต้นด้วย `category` ทำให้โมเดลตัดสินใจไปแล้วตั้งแต่ token ของ category
   พอถึง incident_type ก็แทบแน่นอนแล้ว สัญญาณ "เหตุหรือไม่ใช่เหตุ" ที่ใช้ได้จริงจึงอยู่ที่ P(category = null) ซึ่ง gate เลือกใช้
   ลำดับ key ใน output จึงมีผลต่อการออกแบบ confidence ด้วย
2. **eval_loss หลอกตา:** checkpoint ที่ eval_loss ดีที่สุดกลับมี miss สูงกว่า base เพราะ val เบ้ไปทาง `no_incident`
   ต้องเชื่อ eval ตามงาน (macro-F1 / miss) ไม่ใช่ loss
3. **torchao ที่ติดมากับ Colab ทำ peft พัง:** torchao 0.10 ทำให้ peft 0.19 ขึ้น ImportError ตอนสร้าง LoRA layer
   แก้โดย `pip uninstall -y torchao` ใน bootstrap และนอกจากนี้ `apply_chat_template(images=...)` ใช้ไม่ได้ใน transformers 5.13.1 จึงต้องใส่ภาพใน messages แทน
4. **JPEG encode ซ้ำเปลี่ยนคำตอบ ~5%:** API รุ่นแรกย่อภาพแล้ว encode เป็น JPEG q95 อีกรอบ ผลคือคำตอบต่างจากสคริปต์ 13/247 ภาพ
   ตรวจแล้ว 11 ภาพเกิดจากการ encode ซ้ำ จึงเปลี่ยนเป็น PNG (lossless) และความตรงกันขึ้นเป็น 240/247 · โมเดลเล็กไวต่อ artifact ของภาพมากกว่าที่คิด
5. **runner ค้าง:** `colab exec` ค้างได้ตลอดไปเมื่อ connection หลุด (เกิดตอน poll และตอน pack/download) ถ้าค้างตอนเทรนหลายชั่วโมงอาจเสีย adapter
   จึงครอบทุกคำสั่ง colab ด้วย `timeout`, retry ตอน fetch, มี fallback ที่ดึงแค่ adapter + log และ trap ที่สั่ง `colab stop` เสมอ
   รอบ predict lora_v1 ที่ connection หลุดตอน bootstrap ก็จบได้เองใน ~17 นาทีแทนที่จะค้างจนเผาเครดิต
6. **Qwen3.5 ไม่ใช่ transformer ล้วน:** ชั้น linear attention ใช้ชื่อ module ต่างออกไป ต้องตรวจชื่อ layer จริงก่อนตั้ง `target_modules`
   และตรวจหลัง merge ว่าน้ำหนักของชั้นเหล่านั้นเปลี่ยนจริง
7. **tied embeddings ไม่เป็นปัญหาบน llama.cpp b10909:** converter ของ llama.cpp ใช้ `token_embd` ซ้ำได้เอง
   แต่ `llama-mtmd-cli` ใช้กับ template นี้ไม่ได้ดี จึงใช้ `llama-server` เป็นทางหลัก
8. **เลือก threshold บนชุดที่ไม่ใช้รายงานผล:** เลือก T บน val แล้วรายงานบน gold กับ test ที่ตัด gold ออก
   ถ้าเลือกบน gold ตัวเลขจะสวยเกินจริง

---

## 9. งานต่อ (v2)

- **ข้อมูลไทย:** เก็บภาพแจ้งเหตุจริงจากไทย (ขออนุญาตและลบข้อมูลส่วนบุคคล) อย่างน้อยเป็น**ชุดทดสอบ** ก่อนตัดสินใจใช้งาน
- **ข้อมูลไลเซนส์สะอาด:** แทน CrisisMMD และชุดที่ไม่ระบุไลเซนส์ด้วยข้อมูลที่ใช้เชิงพาณิชย์/ภาครัฐได้ แล้วเทรนใหม่
- **ลด miss ที่ต้นทาง:** ถ่วงน้ำหนักหรือลด `no_incident` ใน train, เลือก checkpoint ด้วย miss หรือ macro-F1 บนชุด val ที่สมดุลแทน eval_loss
- **คลาสหายาก:** เพิ่ม smoke_detected, vehicle_fire, landslide, water_rescue และตัวอย่าง `unsure` ที่เป็นเหตุนอก 12 ประเภทจริง ๆ
- **gold ที่แข็งแรงขึ้น:** ผู้ตรวจ 2 คนแบบไม่เห็น label เดิม วัด agreement และขยายคลาสหายากให้ได้ ≥ 30 ภาพ
- **calibration:** temperature scaling บน val แทน threshold ระดับ 1e-5, หรือเรียงลำดับ key ให้ได้ confidence ที่มีความหมายมากขึ้น
- **วัด GGUF บน test เต็ม** ทั้งแบบ Q8/Q4 และ bf16 ใน runtime เดียวกัน เพื่อแยกผลของ quantization กับ runtime ออกจากกัน
- ทดสอบความทนทาน: ภาพบีบอัดหลายรอบ ภาพหน้าจอ ภาพมืด/ฝน และภาพที่สร้างด้วย AI

---

*ไฟล์อ้างอิงหลัก:* `reports/base-gold.json`, `reports/base-test.json`, `reports/lora_v1-gold.json`, `reports/lora_v1-test.json`,
`reports/gguf_q8-gold.json`, `reports/gguf_q4-gold.json`, `reports/abstain-q4.md`, `README.md`, `PLAN.md`
