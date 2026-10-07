# disaster-vlm-lab

## Data notes (v1)

- **Rare test classes (<30 images):** building_collapse 29, water_rescue 18, building_fire 16, landslide 5, smoke_detected 4, vehicle_fire 2, unsure 2. This is a data limit, not a bug; metrics on these classes are very noisy. Val is smaller still (e.g. landslide 2).
- **Skewed val/test:** only train is balanced, so val/test are about 74% `no_incident`. Evaluate with macro-F1 and per-class metrics, not accuracy.
- **Teacher labels:** about 29% of test labels (947/3244) come from the teacher (`label_source == "teacher"`); the rest are fixed from source mappings. Rows the teacher dropped (answer outside candidates or confidence < 0.5) are excluded. Report metrics split by `label_source`. Every split row carries a `label_source` field.
- **Teacher compute cap:** the full teacher run was capped at 9,500 calls. CrisisMMD multi-candidate rows were subsampled to 5,594 of 11,121 (seed 20261007); all other multi-candidate rows and all severity-only rows were kept.
- **Ordering:** `train.jsonl` is ordered by class (balance output), so the trainer must shuffle.
