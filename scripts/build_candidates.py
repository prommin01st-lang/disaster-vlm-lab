"""ดึง dataset จาก HF → ภาพ 512px + candidates.jsonl → อัปขึ้น HF private dataset repo"""
import hashlib, json, os, sys, tarfile
from pathlib import Path

from datasets import load_dataset
from PIL import Image as PILImage
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


def rows_of(ds):
    """วนแถวโดยไม่ decode ภาพใน datasets (ไฟล์เสียจะได้ไม่ทำให้ทั้ง job ล้ม) — decode เองแล้ว image=None ถ้าเสีย"""
    import io
    from datasets import Image as HFImage
    ds = ds.cast_column("image", HFImage(decode=False))
    for r in ds:
        im = r["image"]
        try:
            if im.get("bytes"):
                r["image"] = PILImage.open(io.BytesIO(im["bytes"])); r["image"].load()
            else:
                r["image"] = PILImage.open(im["path"]); r["image"].load()
        except Exception as e:
            print("skip undecodable image:", e)
            r["image"] = None
        yield r


# 1) CrisisMMD — join informative + damage ด้วย image_id
# คอลัมน์ image ใน repo นี้เป็น None ทุกแถว (ภาพอยู่เป็นไฟล์ที่ data_image/…, อ้างด้วย image_path) → ดึงด้วย snapshot_download
from huggingface_hub import snapshot_download
from PIL import Image as PILImage
CMMD_DIR = Path(snapshot_download("QCRI/CrisisMMD", repo_type="dataset", allow_patterns=["data_image/*"]))


def cmmd_image(r):
    if r["image"] is not None:
        return r["image"]
    p = CMMD_DIR / r["image_path"]
    if not p.exists():
        print("skip missing crisismmd image:", r["image_id"], r["image_path"])
        return None
    try:
        im = PILImage.open(p)
        im.load()
        return im
    except Exception as e:
        print("skip corrupt crisismmd image:", r["image_id"], e)
        return None


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
        add("crisismmd", r["image_id"], cmmd_image(r), cands, sev)
print("crisismmd", len(rows))

# 2) DisasterVQA — 1 ภาพหลายคำถาม: key ด้วย image_id (หรือ hash ภาพถ้า id ว่าง)
for r in rows_of(load_dataset("anwan/DisasterVQA", split="train")):
    sid = r["image_id"] or hashlib.sha1((r["image"].tobytes() if r["image"] is not None else b"")).hexdigest()
    add("disastervqa", sid, r["image"], mapping.disastervqa(r["disaster_type"]), None)
print("+disastervqa", len(rows))

# 3) wildfire
for split in ("train", "validation", "test"):
    ds = load_dataset("AbdullahImran/balanced_wildfire_dataset", split=split)
    names = ds.features["label"].names
    for i, r in enumerate(rows_of(ds)):
        add("wildfire", f"{split}-{i}", r["image"], *mapping.wildfire(names[r["label"]]))
print("+wildfire", len(rows))

# 4) hard negatives
for i, r in enumerate(rows_of(load_dataset("fireviewer/fire_and_smoke_detection_very_hard_negative",
                                   "existing_negative_fire_pics", split="train"))):
    add("hardneg", i, r["image"], ("no_incident",), "none")

# 5) traffic accidents — cap 800 ต่อคลาส (มุม CCTV ซ้ำกันเยอะ)
cnt = {True: 0, False: 0}
tds = load_dataset("hiennguyen9874/traffic-accident-detection", split="train")
for i, acc in enumerate(tds["is_accident"]):  # อ่านคอลัมน์ label ก่อน จะได้ไม่ decode ภาพที่เกิน cap
    acc = bool(acc)
    if cnt[acc] >= 800:
        continue
    cnt[acc] += 1
    r = next(rows_of(tds.select([i])))
    add("traffic", r["image_id"], r["image"], mapping.traffic(acc), "none" if not acc else None)

# 6) pothole
for split in ("train", "validation", "test"):
    ds = load_dataset("Arpitraj01/Pothole_classification", split=split)
    names = ds.features["label"].names
    for i, r in enumerate(rows_of(ds)):
        add("pothole", f"{split}-{i}", r["image"], *mapping.pothole(names[r["label"]]))

# 7) ภาพชีวิตประจำวัน — repo นี้เป็น dataset script (datasets รุ่นใหม่ไม่รองรับ) → อ่านจาก parquet ที่ HF แปลงไว้ (refs/convert/parquet)
def flickr_ds():
    from huggingface_hub import hf_hub_download
    f = hf_hub_download("nlphuji/flickr_1k_test_image_text_retrieval", "TEST/test/0000.parquet",
                        repo_type="dataset", revision="refs/convert/parquet")
    return load_dataset("parquet", data_files=f, split="train", columns=["image"])


try:
    for i, r in enumerate(rows_of(flickr_ds())):
        add("flickr", i, r["image"], ("no_incident",), "none")
except Exception as e:  # แหล่งนี้ใช้ไม่ได้ → ข้ามแต่ log ชัด ๆ ไม่ให้ทั้ง build ล้ม
    print("SKIP SOURCE flickr:", repr(e))
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
