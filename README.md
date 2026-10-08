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

**อัตรา teacher label ผิด (Task 10):** ผู้ใช้ตรวจ 247 ภาพ ไม่แก้เลย 0/176 (teacher) และ 0/71 (fixed) — เป็นการยอมรับ ไม่ใช่หลักฐานว่า teacher ไม่ผิด

**CU ที่ใช้ (Task 12):** เทรนรวม smoke ≈ 5.8 CU, predict lora_v1 ≈ 2.7 CU (รวมรอบที่ connection หลุดตอน bootstrap ~0.4 CU) — L4 ≈ 1.54 CU/ชม.

## ข้อจำกัด

- **ไม่มีภาพจากไทย** — ภาพทั้งหมดมาจาก dataset ต่างประเทศ (CrisisMMD, DisasterVQA ฯลฯ) ผลบนภาพไทยจริงยังไม่รู้
- **คลาสที่ test < 30 ภาพ** (building_collapse 29, water_rescue 18, building_fire 16, landslide 5, smoke_detected 4, vehicle_fire 2, unsure 2) — F1 ของคลาสเหล่านี้แกว่งมาก
- **ไลเซนส์ NC** — dataset ต้นทางบางชุดเป็น non-commercial จึงใช้ได้เพื่อการเรียน/วิจัยเท่านั้น
- label ~29% ของ test มาจาก teacher (Qwen3.5-9B) ไม่ใช่คน
