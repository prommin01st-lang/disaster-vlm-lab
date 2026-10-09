# disaster-vlm-lab

## เป้าหมาย

Fine-tune **Qwen3.5-2B** (VLM) ด้วย LoRA ให้ดูภาพแจ้งเหตุแล้วตอบ JSON บรรทัดเดียว
`{"category": ..., "incident_type": ..., "severity": ...}` ตาม incident catalog ของ SOS — เทรน/predict บน Colab L4
(เครื่อง local 6GB รันไม่ไหว) prompt อยู่ที่เดียวใน `dvl/prompt.py`

**ชุดประเภทเฟส 1** (`dvl/catalog.py`): fire_hazard — `forest_fire`, `building_fire`, `vehicle_fire`, `smoke_detected` ·
disaster — `flood`, `storm`, `landslide`, `damaged_structure`, `road_hazard` · rescue — `building_collapse`, `water_rescue` ·
accident — `vehicle_collision` · เพิ่ม `unsure` (other) และ `no_incident` (category `null`)

## วิธีรัน

ทุกงานโมเดลรันผ่าน `colab/run_job.sh <job.sh> [GPU] [timeout_min]` (bundle → bootstrap → launch → poll → ดึง `out/` มาที่ `runs/dvl-<job>-<เวลา>/` → `colab stop` เสมอ) ·
token HF ส่งเป็นไฟล์ ไม่ผ่าน argv · ทุกคำสั่ง colab มี `timeout` กันค้าง · ดึงผลไม่ได้จะดึงชุดเล็ก (job.log + adapter) ก่อนหยุด VM

```bash
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
colab/run_job.sh colab/jobs/build_candidates.sh CPU 120                 # 1) รวมภาพจาก 7 dataset → Petanque/dvl-data
colab/run_job.sh colab/jobs/teacher.sh L4 420                           # 2) teacher (Qwen3.5-9B) ติด label + แบ่ง train/val/test
../iron-coach-th/.venv/bin/python scripts/review_sheet.py ...           # 3) คนตรวจ test_gold (247 ภาพ)
SPLIT=test TAG=base colab/run_job.sh colab/jobs/predict.sh L4 170        # 4) baseline
MAX_STEPS=20 colab/run_job.sh colab/jobs/train.sh L4 40                  # 5a) smoke เทรน 20 step
colab/run_job.sh colab/jobs/train.sh L4 300                              # 5b) เทรนเต็ม → Petanque/dvl-qwen35-2b-lora
ADAPTER=Petanque/dvl-qwen35-2b-lora TAG=lora_v1 SPLIT=test \
  colab/run_job.sh colab/jobs/predict.sh L4 170                          # 6) predict ด้วย adapter
PYTHONPATH=. ../iron-coach-th/.venv/bin/python scripts/evaluate.py --name gold \
  data/dataset_v1/test_gold.jsonl runs/<base>/out/predictions-base.jsonl runs/<lora>/out/predictions-lora_v1.jsonl
PYTHONPATH=. ../iron-coach-th/.venv/bin/python scripts/evaluate.py --name test \
  data/dataset_v1/test.jsonl runs/<base>/out/predictions-base.jsonl runs/<lora>/out/predictions-lora_v1.jsonl
```

## Data notes (v1)

- **Rare test classes (<30 images):** building_collapse 29, water_rescue 18, building_fire 16, landslide 5, smoke_detected 4, vehicle_fire 2, unsure 2. This is a data limit, not a bug; metrics on these classes are very noisy. Val is smaller still (e.g. landslide 2).
- **Skewed val/test:** only train is balanced, so val/test are about 74% `no_incident`. Evaluate with macro-F1 and per-class metrics, not accuracy.
- **Teacher labels:** about 29% of test labels (947/3244) come from the teacher (`label_source == "teacher"`); the rest are fixed from source mappings. Rows the teacher dropped (answer outside candidates or confidence < 0.5) are excluded. Report metrics split by `label_source`. Every split row carries a `label_source` field.
- **Teacher compute cap:** the full teacher run was capped at 9,500 calls. CrisisMMD multi-candidate rows were subsampled to 5,594 of 11,121 (seed 20261007); all other multi-candidate rows and all severity-only rows were kept.
- **Ordering:** `train.jsonl` is ordered by class (balance output), so the trainer must shuffle.

## Test review

The user reviewed the 247-row gold sheet on 2026-10-08 and accepted all labels unchanged: 0/176 teacher rows and 0/71 fixed rows were changed or discarded. This is a sign-off without edits, not proof that the teacher is error-free. These rows are stored in `test_gold.jsonl` with `label_source="human"`.

## Baseline

Un-tuned Qwen3.5-2B (bf16, greedy, `enable_thinking=False`), one predictions file (3244 test rows, L4, 2.01 s/img) evaluated against both gold sets. Lead metric is macro-F1 and per-class F1; accuracy is not meaningful alone because test is ~74% `no_incident`. `gold` is the 247-row human-reviewed, class-balanced subset (so its macro-F1 is not comparable to the skewed `test`). Reports: `reports/base-gold.json`, `reports/base-test.json`.

### gold (247 rows, label_source=human)

| metric | base |
|---|---|
| n | 247 |
| json_valid_rate | 1.000 |
| type_accuracy | 0.656 |
| type_macro_f1 | 0.525 |
| category_accuracy | 0.765 |
| severity_accuracy | 0.794 |
| false_alarm_rate | 0.400 |
| miss_rate | 0.018 |
| unsure_rate | 0.000 |
| ece | 0.246 |

Per-type F1 (n):
```
  building_collapse    0.62   (n=25)
  building_fire        0.86   (n=16)
  damaged_structure    0.00   (n=25)
  flood                0.67   (n=25)
  forest_fire          0.86   (n=25)
  landslide            0.29   (n=5)
  no_incident          0.68   (n=25)
  road_hazard          0.56   (n=25)
  smoke_detected       0.00   (n=4)
  storm                0.82   (n=25)
  unsure               0.00   (n=2)
  vehicle_collision    0.91   (n=25)
  vehicle_fire         0.67   (n=2)
  water_rescue         0.41   (n=18)
```

### test (3244 rows)

| metric | base |
|---|---|
| n | 3244 |
| json_valid_rate | 0.997 |
| type_accuracy | 0.551 |
| type_macro_f1 | 0.321 |
| category_accuracy | 0.580 |
| severity_accuracy | 0.588 |
| false_alarm_rate | 0.484 |
| miss_rate | 0.007 |
| unsure_rate | 0.003 |
| ece | 0.324 |

Per-type F1 (n):
```
  building_collapse    0.09   (n=29)
  building_fire        0.56   (n=16)
  damaged_structure    0.05   (n=168)
  flood                0.51   (n=118)
  forest_fire          0.64   (n=163)
  landslide            0.09   (n=5)
  no_incident          0.68   (n=2409)
  road_hazard          0.33   (n=69)
  smoke_detected       0.00   (n=4)
  storm                0.52   (n=143)
  unsure               0.00   (n=2)
  vehicle_collision    0.68   (n=98)
  vehicle_fire         0.25   (n=2)
  water_rescue         0.10   (n=18)
```

Per label_source (test):

| group | metric | base |
|---|---|---|
| fixed | n | 2297 |
| fixed | type_macro_f1 | 0.450 |
| fixed | type_accuracy | 0.540 |
| fixed | false_alarm_rate | 0.498 |
| fixed | miss_rate | 0.004 |
| teacher | n | 947 |
| teacher | type_macro_f1 | 0.469 |
| teacher | type_accuracy | 0.578 |
| teacher | false_alarm_rate | 0.411 |
| teacher | miss_rate | 0.009 |

Reading: macro-F1 is low (0.32 on test) because rare classes score near 0 and `damaged_structure` is almost never predicted correctly; the base model over-alarms on `no_incident` images (false alarm 0.48 on test) while rarely missing real incidents. JSON validity is already ~100%, so fine-tuning has to win on label semantics, not format.

## Run locally (GGUF)

The LoRA adapter merged into Qwen3.5-2B, converted with llama.cpp b10909 (`a2878d30d`), is in the private HF repo
`Petanque/dvl-qwen3.5-2b-gguf`: `dvl-qwen3.5-2b-Q8_0.gguf` (1.87 GiB), `dvl-qwen3.5-2b-Q4_K_M.gguf` (1.19 GiB) and
`mmproj-dvl-qwen3.5-2b-F16.gguf` (0.62 GiB, needed for images). The export runs on a Colab CPU runtime:
`HF_TOKEN=$(cat ~/.cache/huggingface/token) colab/run_job.sh colab/jobs/gguf.sh CPU 120` (`scripts/export_gguf.py`).

```bash
# 1) download into models/gguf/ (gitignored)
../iron-coach-th/.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Petanque/dvl-qwen3.5-2b-gguf', local_dir='models/gguf')"

# 2) start the server (Vulkan build; port 8080 is often taken, so use 8091)
LL="../olmocr-lab/bin/llama-b10909"
LD_LIBRARY_PATH="$LL" "$LL/llama-server" -m models/gguf/dvl-qwen3.5-2b-Q8_0.gguf \
  --mmproj models/gguf/mmproj-dvl-qwen3.5-2b-F16.gguf -ngl 99 -c 4096 -np 1 \
  --jinja --reasoning off --host 127.0.0.1 --port 8091

# 3) in another shell: predict the gold set, then score it
PYTHONPATH=. ../iron-coach-th/.venv/bin/python scripts/predict_gguf.py --tag gguf_q8 --url http://127.0.0.1:8091
PYTHONPATH=. ../iron-coach-th/.venv/bin/python scripts/evaluate.py --name gold \
  data/dataset_v1/test_gold.jsonl runs/gguf/predictions-gguf_q8.jsonl
```

`predict_gguf.py` sends the same messages as `dvl/prompt.py` (system prompt, then image, then instruction) with
`temperature 0`, `max_tokens 64`, `enable_thinking: false` and `logprobs`. Confidence uses `dvl/confidence.py` on the
server's pre-sampling token probabilities. The script fails if any answer contains thinking output.

Gold set (247), RTX 2060 6GB, Vulkan. VRAM is the whole card from nvidia-smi (the desktop uses about 350 MiB):

| metric | base (bf16, HF) | gguf_q8 | gguf_q4 |
|---|---|---|---|
| json_valid_rate | 1.000 | 1.000 | 1.000 |
| type_macro_f1 | 0.525 | 0.690 | 0.689 |
| type_accuracy | 0.656 | 0.798 | 0.798 |
| false_alarm_rate | 0.400 | 0.120 | 0.080 |
| miss_rate | 0.018 | 0.041 | 0.045 |
| ece | 0.246 | 0.160 | 0.158 |
| s/img | 2.01 (L4) | 0.58 | 0.46 |
| VRAM peak (MiB) | – | 3294 | 2543 |

Reports: `reports/gguf_q8-gold.json`, `reports/gguf_q4-gold.json`. Q4_K_M scores the same as Q8_0 within noise on
this small set (per-class differences are 1–2 images), so it is a reasonable default on a 6GB card. The bf16 LoRA
reference on the same set (`lora_v1`) is needed to measure quantization loss directly.

## LoRA v1 (lora_v1) เทียบ base

**การเทรน:** LoRA r=16/alpha=32/dropout 0.05 บน q/k/v/o, gate/up/down และ linear-attention (`in_proj_qkv`, `in_proj_z`, `out_proj`)
ไม่แตะ vision · trainable 15.6M/2.23B (0.70%) · bf16, batch 4×accum 4, LR 1e-4 cosine, 2 epoch = 902 step · loss เฉพาะ token คำตอบ ·
เลือก checkpoint ตาม eval_loss บน val → ดีสุด step 700 (eval_loss 0.0239) · เทรน 3.4 ชม. บน L4
· adapter: `Petanque/dvl-qwen35-2b-lora` (private) · predict 1.70 s/ภาพ · รายงาน: `reports/lora_v1-gold.json`, `reports/lora_v1-test.json`

### gold (247 ภาพ, คนตรวจ)

| metric | base | lora_v1 |
|---|---|---|
| json_valid_rate | 1.000 | 1.000 |
| type_accuracy | 0.656 | 0.794 |
| **type_macro_f1** | 0.525 | **0.670** |
| category_accuracy | 0.765 | 0.858 |
| severity_accuracy | 0.794 | 0.858 |
| **false_alarm_rate** | 0.400 | **0.160** |
| **miss_rate** | 0.018 (4/222) | **0.036 (8/222)** |
| ece | 0.246 | 0.173 |

### test (3244 ภาพ, ~74% no_incident)

| metric | base | lora_v1 |
|---|---|---|
| json_valid_rate | 0.997 | 1.000 |
| type_accuracy | 0.551 | 0.879 |
| **type_macro_f1** | 0.321 | **0.607** |
| category_accuracy | 0.580 | 0.893 |
| severity_accuracy | 0.588 | 0.871 |
| **false_alarm_rate** | 0.484 | **0.117** |
| **miss_rate** | 0.007 (6/835) | **0.025 (21/835)** |
| ece | 0.324 | 0.105 |

แยกตาม label_source (test): fixed macro-F1 0.450→0.704, FA 0.498→0.126, miss 0.004→0.012 · teacher macro-F1 0.469→0.676, FA 0.411→0.068, miss 0.009→0.031

### per-type F1 (base → lora_v1)

| type | gold (n) | test (n) |
|---|---|---|
| building_collapse | 0.62 → 0.61 (25) | 0.09 → 0.58 (29) |
| building_fire | 0.86 → 0.79 (16) | 0.56 → 0.63 (16) |
| damaged_structure | 0.00 → 0.64 (25) | 0.05 → 0.73 (168) |
| flood | 0.67 → 0.83 (25) | 0.51 → 0.76 (118) |
| forest_fire | 0.86 → 0.89 (25) | 0.64 → 0.90 (163) |
| landslide | 0.29 → 0.60 (5) | 0.09 → 0.46 (5) |
| no_incident | 0.68 → 0.78 (25) | 0.68 → 0.93 (2409) |
| road_hazard | 0.56 → 0.80 (25) | 0.33 → 0.68 (69) |
| smoke_detected | 0.00 → 0.00 (4) | 0.00 → 0.00 (4) |
| storm | 0.82 → 0.92 (25) | 0.52 → 0.66 (143) |
| unsure | 0.00 → 0.00 (2) | 0.00 → 0.00 (2) |
| vehicle_collision | 0.91 → 0.98 (25) | 0.68 → 0.91 (98) |
| vehicle_fire | 0.67 → 0.67 (2) | 0.25 → 0.67 (2) |
| water_rescue | 0.41 → 0.88 (18) | 0.10 → 0.59 (18) |

### ผลตามเกณฑ์: **ไม่ผ่าน (miss_rate แย่ลง)**

| เกณฑ์ | gold | test |
|---|---|---|
| macro-F1 สูงกว่า base | ผ่าน (0.525→0.670) | ผ่าน (0.321→0.607) |
| false_alarm ไม่สูงกว่า base | ผ่าน (0.400→0.160) | ผ่าน (0.484→0.117) |
| miss ไม่สูงกว่า base | **ไม่ผ่าน** (0.018→0.036) | **ไม่ผ่าน** (0.007→0.025) |
| json_valid ≥ 0.99 | ผ่าน (1.000) | ผ่าน (1.000) |

อ่านผล: ดีขึ้นชัดเกือบทุกคลาส (damaged_structure, water_rescue, building_collapse จากเกือบ 0) และ false alarm ลดลง 3–4 เท่า
แต่ miss เพิ่ม — base miss ต่ำเพราะแทบไม่เคยตอบ `no_incident` (false alarm 0.48) ส่วน lora_v1 ยอมตอบ `no_incident` มากขึ้น
ภาพเหตุที่พลาดเป็น no_incident บน test (21 ภาพ): storm 6, road_hazard 5, damaged_structure 3, flood 3, forest_fire 2, unsure 2
(gold: 8 ภาพ เพิ่มจาก 4) · `smoke_detected` และ `unsure` ยัง 0 ทั้งคู่ (train มีแค่ 34/20 ภาพ)
· ทางแก้ที่น่าลองใน v2: ถ่วงน้ำหนัก/ลด `no_incident` ในชุดเทรน หรือเลือก checkpoint ด้วย miss-rate แทน eval_loss
(val เป็น no_incident 74% ทำให้ eval_loss เอียงไปทาง no_incident) หรือใช้ threshold confidence ให้ no_incident ต้องมั่นใจสูงก่อนตัดสิน

### กติกาตอนใช้งาน: no_incident ที่ไม่มั่นใจ → `unsure` (ส่งให้คนดู) — GGUF Q4_K_M

ไม่ต้องเทรนใหม่: ถ้าโมเดลตอบ `no_incident` แต่ **P(เป็นเหตุ) = 1 − P(category = null) ≥ T** ให้เปลี่ยนเป็น `unsure`
(category มาก่อน incident_type ใน JSON จึงเป็น token ที่โมเดลตัดสินว่าเหตุ/ไม่ใช่เหตุ — confidence ของ incident_type อิ่มที่ 1.0 ใช้แยกไม่ได้)
`scripts/predict_gguf.py` เก็บ `category_alts`/`type_alts` (top-5 logprobs) · `scripts/abstain_sweep.py` เลือก T บน **val** แล้ววัดบนชุดอื่น
เลือก **T = 3.2e-5** (T ใหญ่สุดที่ val miss ≤ 1%) — ผลเต็มใน `reports/abstain-q4.md`

| ชุด (Q4_K_M) | กติกา | macro-F1 | false_alarm | miss | ส่งให้คนดู (unsure) |
|---|---|---|---|---|---|
| val 1082 (ใช้เลือก T) | ปิด → เปิด | 0.674 → 0.671 | 0.142 → 0.142 | 0.024 → **0.003** | 0% → 9.7% |
| gold 247 | ปิด → เปิด | 0.684 → 0.697 | 0.080 → 0.080 | 0.045 → **0.018** | 0% → 5.3% |
| test ไม่รวม gold 2997 | ปิด → เปิด | 0.724 → 0.715 | 0.130 → 0.130 | 0.031 → **0.003** | 0.1% → 10.4% |

เทียบเกณฑ์กับ base: miss ไม่สูงกว่า base แล้ว (gold 0.018 = 0.018, test 0.003 < 0.007) และ false alarm ยังต่ำกว่า base 3–6 เท่า
ราคา: ~10% ของภาพถูกส่งให้คนดู — บน test 310 ภาพ เป็นเหตุจริง 17 (storm 7, road_hazard 4, …) ที่เหลือ 293 ไม่ใช่เหตุ
ข้อจำกัด: T ผูกกับไฟล์ Q4_K_M + llama.cpp b10909 (Q8_0/bf16 ต้อง sweep ใหม่) · ค่า T เล็กมาก (ระดับ 1e-5) เพราะโมเดลมั่นใจเกินจริง
· logprob มาจาก top-5 เท่านั้น — ถ้า `null` หลุด top-5 จะนับเป็นเหตุ (ปลอดภัย = ส่งให้คนดู)

**อัตรา teacher label ผิด (Task 10):** ผู้ใช้ตรวจ 247 ภาพ ไม่แก้เลย 0/176 (teacher) และ 0/71 (fixed) — เป็นการยอมรับ ไม่ใช่หลักฐานว่า teacher ไม่ผิด

**CU ที่ใช้ (Task 12):** เทรนรวม smoke ≈ 5.8 CU, predict lora_v1 ≈ 2.7 CU (รวมรอบที่ connection หลุดตอน bootstrap ~0.4 CU) — L4 ≈ 1.54 CU/ชม.

## HTTP API

API บนเครื่องให้ระบบ SOS (ASP.NET Core) เรียก: ส่งภาพ 1 ภาพ → ได้ JSON เหตุ + ธง `needs_review`
ห่อ llama-server (GGUF **Q4_K_M** + mmproj) และใช้กติกา abstain ข้างบน (T = 3.2e-5) ทุกครั้ง — โค้ดอยู่ที่ `api/app.py`
(client ของ llama.cpp + กติกา abstain อยู่ที่ `dvl/llamacpp.py` ใช้ร่วมกับ `scripts/predict_gguf.py` / `abstain_sweep.py`)

### เปิดใช้งาน

```bash
../iron-coach-th/.venv/bin/pip install -r api/requirements.txt   # ครั้งแรก
api/run.sh   # เปิด llama-server :8091 (log ใน runs/) ถ้ายังไม่ healthy → รอ /health → uvicorn :8092
             # Ctrl+C ปิดทั้ง uvicorn และ llama-server ที่สคริปต์เปิดเอง (ถ้า llama-server เปิดอยู่ก่อนแล้ว จะใช้ต่อและไม่ปิด)
```

ค่าเริ่มต้น **bind ที่ 127.0.0.1** ทั้งสองพอร์ต (เรียกได้จากเครื่องนี้เท่านั้น) — ถ้าจะให้เครื่องอื่นเรียก ตั้ง `DVL_API_HOST=0.0.0.0` และตั้ง `DVL_API_KEY` ด้วย
กิน VRAM ทั้งการ์ด ~2.5 GB (วัดจริงบน 2060 รวม desktop ~0.3 GB) · ปิดก่อนเทรน/รันงานอื่นที่ใช้ GPU

### Endpoints

| method | path | ทำอะไร |
|---|---|---|
| `POST` | `/v1/classify` | multipart field `image` (jpeg/png/webp ≤ 10 MB และ ≤ 40 ล้าน pixel) → ผลจำแนก |
| `GET` | `/v1/labels` | `incident_type` ทั้ง 14 ค่าที่โมเดลตอบได้ + `category` + `name_th` (จาก `dvl/catalog.py`) |
| `GET` | `/health` | 200 เมื่อ llama-server `/health` ok · ไม่งั้น 503 (กำลังโหลดโมเดล/ไม่ได้เปิด) · ไม่ต้องใช้ key |

ภาพที่ส่งมาถูกแปลงด้วย `to_rgb_resized(512)` (หมุนตาม EXIF, RGBA → พื้นขาว, ด้านยาว ≤ 512 แบบเดียวกับตอนสร้าง dataset) → **PNG (lossless)** → llama-server
(ไม่ encode JPEG ซ้ำ: JPEG รอบสองทำให้คำตอบเปลี่ยน ~5% บน gold — PNG ส่ง pixel เดียวกับที่ถอดจากไฟล์ต้นฉบับ)
คำขอไป llama-server เข้าคิวทีละคำขอ (server เปิด `-np 1`) — ส่งพร้อมกันได้แต่จะรอคิว · ถอดภาพพร้อมกันได้ 2 ภาพ

กันคำขอกิน RAM/ดิสก์: body ทั้งคำขอ ≤ 10 MB + 64 KB (นับระหว่างอ่าน ใช้ได้กับ chunked ที่ไม่มี Content-Length) ·
ภาพที่ header ประกาศเกิน 40 ล้าน pixel ถูกปฏิเสธก่อนถอด (`413 too_many_pixels`) · JPEG ถอดแบบย่อ (draft ≥ 1024) ก่อนย่อเหลือ 512

### คำตอบ (key คงที่)

```json
{"category": null, "incident_type": "no_incident", "model_incident_type": "no_incident",
 "severity": "none", "confidence": 1.0,
 "p_incident": 3.32e-06, "needs_review": false, "review_reason": null,
 "incident_type_name_th": "ไม่ใช่เหตุ", "valid": true, "model": "dvl-qwen3.5-2b-Q4_K_M",
 "threshold": 3.2e-05, "latency_ms": 818}
```

| key | ความหมาย |
|---|---|
| `category` | หมวดใน SOS catalog (`fire_hazard`, `disaster`, `rescue`, `accident`, `other`) · `null` เมื่อ `no_incident` |
| `incident_type` | ค่า **หลัง** ใช้กติกา abstain — หนึ่งใน `/v1/labels` (`no_incident` ไม่มีใน SOS = ไม่ต้องสร้างเหตุ) |
| `model_incident_type` | คำตอบดิบของโมเดล **ก่อน** กติกา abstain (ต่างจาก `incident_type` เมื่อ `review_reason = uncertain_no_incident`) · `null` เมื่อคำตอบพัง |
| `severity` | `none` (เฉพาะ no_incident) / `mild` / `severe` |
| `confidence` | ความน่าจะเป็นของ token ค่า incident_type ที่โมเดลตอบ — อ้างถึง `model_incident_type` (คำตอบดิบ) ไม่ใช่ค่าหลังกติกา (0–1 ปัด 3 ตำแหน่ง; 0.0 ถ้าคำตอบพัง) — มั่นใจเกินจริง อย่าใช้แทน `needs_review` |
| `p_incident` | P(เป็นเหตุ) = 1 − P(category = null) — ค่าที่กติกา abstain ใช้ |
| `needs_review` | `true` = ต้องให้คนดูก่อนเชื่อผล |
| `review_reason` | `null` · `uncertain_no_incident` (โมเดลตอบ no_incident แต่ `p_incident ≥ threshold` → เปลี่ยนเป็น `unsure`/`other`/`mild`) · `model_unsure` (โมเดลตอบ `unsure` เอง) · `invalid_output` (คำตอบไม่ใช่ JSON ที่ถูกต้อง → `unsure`, `valid=false`) |
| `incident_type_name_th` | ชื่อไทยของ `incident_type` |
| `valid` | คำตอบของโมเดลเป็น JSON ที่ถูกต้องหรือไม่ |
| `model` / `threshold` | ชื่อโมเดล (`DVL_MODEL_NAME`) / T ที่ใช้ |
| `latency_ms` | เวลาใน API ทั้งคำขอ (ถอดภาพ + รอคิว + llama-server) — ~450 ms/ภาพบน 2060 |

**`needs_review` คืออะไร:** โมเดลพลาดเหตุจริงโดยตอบ `no_incident` ได้ — กติกา abstain ส่งภาพ no_incident ที่ไม่มั่นใจพอให้คนดูแทน
(gold: miss 0.045 → 0.018 แลกกับ ~5–10% ของภาพที่ต้องดู) ฝั่ง SOS ควรถือ `needs_review=true` = "อาจเป็นเหตุ ให้เจ้าหน้าที่ยืนยัน" ไม่ใช่ทิ้ง

Error เป็น JSON `{"error": "<code>", "message": "..."}`:
`400 missing_image / undecodable_image` · `401 unauthorized` · `413 too_large / too_many_pixels` · `415 unsupported_media_type` (ตรวจทั้ง content-type และเนื้อไฟล์)
· `502 llama_unreachable / llama_error` · `504 llama_timeout`

### ตัวแปร env

| ตัวแปร | ค่าเริ่มต้น | |
|---|---|---|
| `DVL_LLAMA_URL` | `http://127.0.0.1:8091` | llama-server (run.sh ตั้งจาก `DVL_LLAMA_PORT`) |
| `DVL_ABSTAIN_T` | `3.2e-5` | T ของกติกา abstain — ผูกกับ Q4_K_M + llama.cpp b10909 (โมเดลอื่นต้อง sweep ใหม่) |
| `DVL_MODEL_NAME` | `dvl-qwen3.5-2b-Q4_K_M` | ชื่อใน `model` (run.sh ใช้ชื่อไฟล์ GGUF) |
| `DVL_API_KEY` | (ไม่ตั้ง = ปิด) | ถ้าตั้ง ทุก `/v1/*` ต้องมี header `X-API-Key` ไม่งั้น 401 |
| `DVL_TIMEOUT` | `60` | วินาทีที่รอ llama-server ต่อคำขอ (เกิน → 504) |
| `DVL_MODEL`, `DVL_MMPROJ` | `models/gguf/…Q4_K_M.gguf`, `…/mmproj-…F16.gguf` | (run.sh) ไฟล์โมเดล |
| `DVL_LLAMA_BIN` | `../olmocr-lab/bin/llama-b10909` | (run.sh) โฟลเดอร์ llama-server |
| `DVL_LLAMA_PORT` / `DVL_API_HOST` / `DVL_API_PORT` | `8091` / `127.0.0.1` / `8092` | (run.sh) |

### ตัวอย่าง

```bash
curl -s http://127.0.0.1:8092/health
curl -s -H "X-API-Key: $DVL_API_KEY" -F "image=@photo.jpg" http://127.0.0.1:8092/v1/classify
```

เรียกจาก ASP.NET Core (`HttpClient` จาก `IHttpClientFactory`) — **`HttpClient.Timeout` ต้องมากกว่า `DVL_TIMEOUT` + เวลารอคิว**:
`DVL_TIMEOUT` นับเฉพาะตอนรอ llama-server ไม่นับเวลารอคิว (คำขอละ ~0.5 s × จำนวนคำขอที่รออยู่ก่อนหน้า) เช่นตั้ง 120 s:

```csharp
using var form = new MultipartFormDataContent();
var file = new StreamContent(imageStream);
file.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");   // jpeg/png/webp เท่านั้น
form.Add(file, "image", "photo.jpg");                               // ชื่อ field ต้องเป็น "image"
using var req = new HttpRequestMessage(HttpMethod.Post, "http://127.0.0.1:8092/v1/classify") { Content = form };
req.Headers.Add("X-API-Key", apiKey);                                // ถ้าตั้ง DVL_API_KEY
using var res = await http.SendAsync(req, ct);
res.EnsureSuccessStatusCode();
var result = await res.Content.ReadFromJsonAsync<ClassifyResult>(
    new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower }, ct);
// result.NeedsReview → ส่งให้เจ้าหน้าที่ยืนยัน · IncidentType == "no_incident" && !NeedsReview → ไม่ใช่เหตุ
```

### Bruno collection

`bruno/` — เอกสาร contract ของทุก endpoint (docs ภาษาไทย + ตัวอย่าง request/response จริงทุก status) และชุดเทสที่รันได้
(fixtures ของ 01–03 เป็นภาพจริงจาก test_gold รวมภาพที่ทำให้กติกา abstain ทำงาน — **ไม่แจกใน repo** (ลิขสิทธิ์ของเจ้าของภาพ)
ดึงด้วย `bruno/fetch_fixtures.py` จาก HF dataset private หรือ dataset บนเครื่อง; ยังไม่ดึง → 01–03 skip):

```bash
HF_TOKEN=hf_... python bruno/fetch_fixtures.py               # ครั้งแรก: stream dataset_v1.tar แตกเฉพาะ 4 ภาพ
python bruno/fetch_fixtures.py --from-local data/dataset_v1  # หรือจาก dataset บนเครื่อง
cd bruno
bru run --env Local                              # API เปิดแบบไม่มี key (06-Wrong API Key ถูก skip)
bru run --env Local --env-var apiKey=secret123   # API เปิดด้วย DVL_API_KEY=secret123
```

`08-Too Large` ถูก skip เสมอ (ต้องใช้ไฟล์ > 10 MB ที่ไม่ได้ commit — ดู docs ในไฟล์)

### ตรวจจริงบนเครื่อง (gold 247 ภาพ ผ่าน `api/run.sh`)

| | API (PNG lossless) | API (รอบแรก, JPEG q95) | `predict_gguf.py` + กติกา (ส่งไฟล์ดิบ) |
|---|---|---|---|
| incident_type ตรงกับสคริปต์ | 240/247 (97.2%) | 234/247 (94.7%) | – |
| ตอบถูกตาม gold | 196 | 192 | 191 |
| miss (เหตุจริง → no_incident) | 4/222 | 3/222 | 4/222 |
| false alarm (no_incident → เหตุ) | 2/25 | 2/25 | 2/25 |
| ส่งให้คนดู (gate) | 11 | 14 | 13 |
| latency p50 / p90 / max (ms) | 503 / 553 / 926 | 444 / 490 / 792 | – |

PNG: ต่างจากสคริปต์ 7 ภาพ ทุกภาพเป็นเหตุจริงที่ API ตอบเป็นเหตุ (2 ภาพที่สคริปต์ตอบ no_incident แล้ว gate ส่งคนดู API ตอบเหตุตรง ๆ, อีก 5 ภาพสลับประเภทเหตุ)
— pixel เท่ากับไฟล์ดิบ ส่วนที่ต่างน่าจะมาจากตัวถอด JPEG ต่างกัน (PIL vs stb_image ใน llama.cpp) และ server ไม่ deterministic ข้ามรอบ
(รอบ JPEG: 11/13 ภาพที่ต่างเกิดจาก encode ซ้ำ ยืนยันโดยส่งไฟล์ดิบ vs re-encode ตรงเข้า llama-server)
· PNG ช้ากว่า ~60 ms/ภาพ (encode + payload ใหญ่ขึ้น) · VRAM peak 2579 MiB (ทั้งการ์ด)

## Docker

รัน API + llama-server เป็น container (`docker-compose.yml`) — ไม่ต้องมี venv / llama.cpp บนเครื่อง
มี 2 service หลัก: **`llama`** (image ทางการ `ghcr.io/ggml-org/llama.cpp` เปิด GGUF Q4_K_M + mmproj ด้วย flag เดียวกับ `api/run.sh`)
และ **`api`** (`docker/Dockerfile.api`: python:3.12-slim + FastAPI/Pillow เท่านั้น ไม่มี torch, รันเป็น user ไม่ใช่ root)
บวก one-shot **`models`** ที่ดาวน์โหลด GGUF จาก HF ถ้ายังไม่มี

### ต้องมี

- Docker Engine + Compose v2.20 ขึ้นไป (ใช้ `depends_on.required`)
- profile `cuda`: GPU NVIDIA + **[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)**
  (`nvidia-ctk runtime configure --runtime=docker` แล้ว restart docker) — ไม่มีก็ใช้ profile `cpu`
- ไฟล์ GGUF 2 ไฟล์ใน `models/gguf/` (หรือ `DVL_MODELS_DIR`) — มีแล้ว mount ใช้ได้เลย, ไม่มีให้ service `models` โหลดจาก
  HF repo private `Petanque/dvl-qwen3.5-2b-gguf` ด้วย `HF_TOKEN` (ตรวจ sha256 ก่อนใช้)

### ตั้งค่า (.env)

```bash
cp .env.example .env    # .env ถูก gitignore — ห้าม commit
mkdir -p models/gguf    # ถ้ายังไม่มี (ให้ Docker สร้างเองจะได้โฟลเดอร์ของ root แล้ว service models เขียนไม่ได้)
```

| ตัวแปร | ค่าเริ่มต้น | |
|---|---|---|
| `COMPOSE_PROFILES` | `cpu` | profile ตอนสั่ง `docker compose up` เฉย ๆ (`cpu` / `cuda`) |
| `HF_TOKEN` | (ว่าง) | ใช้เฉพาะ service `models` ตอนไฟล์ขาด — ส่งเป็น env ตอนรัน ไม่ถูก bake ลง image |
| `DVL_API_KEY` | (ว่าง = ปิด) | เหมือน `api/run.sh` — **ตั้งเสมอ** ถ้า `DVL_API_BIND` ไม่ใช่ `127.0.0.1` |
| `DVL_API_BIND` / `DVL_API_PORT` | `127.0.0.1` / `8092` | พอร์ตบน host ของ API (พอร์ตเดียวที่เปิดออกนอก Docker) |
| `DVL_MODELS_DIR` | `./models/gguf` | โฟลเดอร์ GGUF บน host (llama mount แบบ read-only) |
| `DVL_UID` / `DVL_GID` | `1000` / `1000` | user ที่ service `models` ใช้เขียนไฟล์ลง `DVL_MODELS_DIR` |
| `DVL_ABSTAIN_T` / `DVL_TIMEOUT` | `3.2e-5` / `60` | ส่งต่อให้ API (ดูตาราง env ของ HTTP API) |

### เปิดใช้งาน

```bash
docker compose --profile cpu  up -d --build   # CPU (ไม่ต้องมี GPU)
docker compose --profile cuda up -d --build   # GPU NVIDIA (-ngl 99)
docker compose ps                             # รอ llama + api เป็น (healthy) — llama โหลดโมเดลไม่กี่วินาที
curl -s http://127.0.0.1:8092/health          # {"status":"ok","llama":"ok",...}
docker compose --profile cpu down             # ปิด (ไฟล์โมเดลอยู่ใน models/gguf ไม่หาย)
```

ลำดับ: `models` (จบด้วย exit 0 ถ้ามีไฟล์ครบ / โหลดเสร็จ — ไฟล์ขาดและไม่มี token → exit 1 และ llama ไม่เริ่ม)
→ `llama-cpu` หรือ `llama-cuda` (healthcheck `/health` ของ llama-server, network alias `llama:8080`) → `api` (`DVL_LLAMA_URL=http://llama:8080`, healthcheck `/health` ของ API)
llama-server ไม่เปิดพอร์ตออก host — เข้าถึงได้จาก network ของ compose เท่านั้น · Bruno (`bruno/`) รันกับ `http://127.0.0.1:8092` ได้เหมือน `api/run.sh`

### ตรวจจริงบนเครื่อง (profile cpu, Ryzen 7 4800H 16 thread)

- `--profile cpu up -d --build` → healthy ทั้งคู่ · `/health` 200, `/v1/labels` 14 ค่า · Bruno: 9 ผ่าน + 2 skip (06 ไม่มี key, 08 เสมอ) / 23 tests ผ่าน
- latency ต่อภาพ **~3–5 s บน CPU** (GPU Vulkan ~0.5 s) — ผล fixtures ตรงกับ GPU: flood/building_fire ถูก, ภาพ no_incident `p_incident` 4.0e-6 (GPU 4.8e-6) ไม่ส่งคนดู,
  ภาพโปสเตอร์พายุ `p_incident` 1.4e-4 (GPU 1.2e-4) → gate ทำงานเป็น `unsure`
- profile `cuda` ตรวจแค่ `docker compose --profile cuda config` (เครื่องนี้ไม่มี nvidia-container-toolkit) — ยังไม่ได้รันจริง
- image `api` ~233 MB (import แค่ `dvl.{catalog,confidence,imgutil,llamacpp,prompt,schema}` ไม่มี torch/numpy)

### ข้อควรระวัง: threshold ผูกกับ build

`DVL_ABSTAIN_T = 3.2e-5` tune บน val ด้วย **Q4_K_M + llama.cpp b10909 (Vulkan)** แต่ไม่มี image Docker ของ b10909 —
compose pin **`server-b10902` / `server-cuda-b10902`** (build ที่ใกล้ที่สุด; ตรวจด้วย `llama-server --version` → `build 10902, commit df03399b8`)
และ backend ต่างกัน (CPU / CUDA vs Vulkan) ทำให้ logprob ระดับ 1e-5 ขยับได้ → อัตราที่ gate ส่งให้คนดูอาจต่างจากตัวเลขใน README เล็กน้อย
ถ้าจะใช้จริงด้วย profile ไหน ให้ sweep T ใหม่กับ server ตัวนั้น (`scripts/abstain_sweep.py` ชี้ไปที่ llama-server ใน container) แล้วตั้ง `DVL_ABSTAIN_T` ·
เปลี่ยน tag ของ image หรือ quant ก็ต้อง sweep ใหม่เช่นกัน

### ให้ SOS backend เรียก

- SOS รันบน host เดียวกัน (ไม่ใช่ container): `http://127.0.0.1:8092`
- SOS รันใน Docker: ต่อ container ของ SOS เข้า network ของ compose นี้ (`disaster-vlm-lab_default`) แล้วเรียก **`http://api:8092`**
  (ไม่ต้องเปิดพอร์ตออก host) เช่นใน compose ของ SOS:

  ```yaml
  services:
    backend:
      networks: [default, dvl]
  networks:
    dvl:
      name: disaster-vlm-lab_default
      external: true
  ```

- SOS อยู่คนละเครื่อง: ตั้ง `DVL_API_BIND=0.0.0.0` **และ** `DVL_API_KEY` (ส่ง header `X-API-Key`) — ควรมี reverse proxy/TLS ข้างหน้า
- CPU ช้ากว่า GPU ~10 เท่า: ตั้ง `HttpClient.Timeout` ของ SOS ให้เผื่อคิว (คำขอละหลายวินาที × จำนวนที่รอ)

## ข้อจำกัด

- **ไม่มีภาพจากไทย** — ภาพทั้งหมดมาจาก dataset ต่างประเทศ (CrisisMMD, DisasterVQA ฯลฯ) ผลบนภาพไทยจริงยังไม่รู้
- **คลาสที่ test < 30 ภาพ** (building_collapse 29, water_rescue 18, building_fire 16, landslide 5, smoke_detected 4, vehicle_fire 2, unsure 2) — F1 ของคลาสเหล่านี้แกว่งมาก
- **ไลเซนส์ NC** — dataset ต้นทางบางชุดเป็น non-commercial จึงใช้ได้เพื่อการเรียน/วิจัยเท่านั้น
- label ~29% ของ test มาจาก teacher (Qwen3.5-9B) ไม่ใช่คน
