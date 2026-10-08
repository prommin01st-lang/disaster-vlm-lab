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
