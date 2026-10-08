# disaster-vlm-lab

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
